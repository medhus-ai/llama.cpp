#include "llama-moe-pool.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "gguf.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/stat.h>
#include <stdexcept>

static const char * SLOT_IDS_PREFIX = "ffn_moe_slot_ids-";
static const char * PREFETCH_IDS_PREFIX = "ffn_moe_prefetch_ids-";
static const char * PREFETCH_ENTRY_IDS_PREFIX = "ffn_moe_prefetch_entry_ids-";

llama_moe_pool::llama_moe_pool(llama_model & model, int32_t n_slots_, int32_t store_mode, const std::string & model_path, bool verify, int32_t io_threads, uint64_t host_cache_bytes,
                               int32_t prefetch_k)
    : n_slots(n_slots_) {
    if (n_slots <= 0) {
        throw std::runtime_error("moe pool: n_slots must be > 0");
    }
    const uint32_t n_expert = model.hparams.n_expert;
    if (n_expert == 0) {
        throw std::runtime_error("moe pool: model has no routed experts");
    }
    index_.n_expert = n_expert;

    // --- adapter: discover routed-expert tensors per layer (generic over llama_layer fields) ---
    struct kind_t { const char * name; ggml_tensor * llama_layer::* field; };
    const kind_t kinds[] = {
        { "gate", &llama_layer::ffn_gate_exps },
        { "up",   &llama_layer::ffn_up_exps   },
        { "down", &llama_layer::ffn_down_exps },
    };

    size_t n_tensors = 0;
    for (uint32_t il = 0; il < model.layers.size(); ++il) {
        auto & L = model.layers[il];
        bool any = false;
        for (const auto & k : kinds) {
            if (L.*k.field) { any = true; }
        }
        if (!any) {
            continue;
        }
        if (L.ffn_gate_up_exps) {
            throw std::runtime_error("moe pool: merged gate_up expert tensors are not supported yet (layer " + std::to_string(il) + ")");
        }
        if (L.ffn_gate_exps_b || L.ffn_up_exps_b || L.ffn_down_exps_b ||
            L.ffn_gate_exps_s || L.ffn_up_exps_s || L.ffn_down_exps_s) {
            throw std::runtime_error("moe pool: per-expert bias/scale tensors are not supported yet (layer " + std::to_string(il) + ")");
        }
        index_.moe_layers.push_back(il);
        for (const auto & k : kinds) {
            if (L.*k.field) { n_tensors++; }
        }
    }
    if (index_.moe_layers.empty()) {
        throw std::runtime_error("moe pool: no expert tensors found");
    }

    // --- pool tensors: one per (layer, kind), ne[2] = n_slots, same buffer type as the source ---
    // a ggml context per buffer type; tensors are created in the context of the layer's device buft
    auto ctx_for = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_by_buft_.find(buft);
        if (it != ctx_by_buft_.end()) {
            return it->second;
        }
        ggml_init_params ip = { ggml_tensor_overhead() * n_tensors, nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (!c) {
            throw std::runtime_error("moe pool: ggml_init failed");
        }
        ctx_by_buft_[buft] = c;
        return c;
    };

    // When reading from the file, resolve every expert tensor's absolute offset from the GGUF metadata.
    // The index stays model-independent: it only records byte ranges.
    gguf_context * meta = nullptr;
    uint64_t       meta_data_offset = 0;
    if (store_mode >= 1) {
        if (model_path.empty()) {
            throw std::runtime_error("moe pool: file store requested but the model path is unknown (split models are not supported yet)");
        }
        gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        meta = gguf_init_from_file(model_path.c_str(), gp);
        if (!meta) {
            throw std::runtime_error("moe pool: cannot read GGUF metadata from " + model_path);
        }
        meta_data_offset = gguf_get_data_offset(meta);
    }

    ggml_backend_buffer_type_t buft = nullptr;
    index_.descs.resize(index_.moe_layers.size() * n_expert);


    // Cap the pool at the number of distinct experts it could ever hold: n_expert per layer, or
    // n_expert * n_moe_layers when one pool serves every layer.
    {
        const uint64_t cap = n_expert;
        if ((uint64_t) n_slots > cap) {
            fprintf(stderr, "moe pool: %d slots exceeds the %llu experts a layer has, clamping\n",
                    n_slots, (unsigned long long) cap);
            n_slots = (int32_t) cap;
        }
    }

    for (size_t ls = 0; ls < index_.moe_layers.size(); ++ls) {
        const uint32_t il = index_.moe_layers[ls];
        auto & L = model.layers[il];
        std::vector<ggml_tensor *> pool_tensors;
        std::vector<ggml_tensor *> sources;
        for (const auto & k : kinds) {
            ggml_tensor * src = L.*k.field;
            if (!src) {
                continue;
            }
            if (src->ne[2] != (int64_t) n_expert) {
                throw std::runtime_error(std::string("moe pool: unexpected expert axis in ") + ggml_get_name(src));
            }
            if (!ggml_is_contiguous(src)) {
                throw std::runtime_error(std::string("moe pool: expert tensor not contiguous: ") + ggml_get_name(src));
            }
            const char * src_buft_name = ggml_backend_buft_name(ggml_backend_buffer_get_type(src->buffer));
            if (strstr(src_buft_name, "REPACK") != nullptr) {
                throw std::runtime_error(std::string("moe pool: expert tensors are in the ") + src_buft_name +
                                         " buffer type, whose contents cannot be read back; rerun with --no-repack");
            }
            // the pool lives where the layer computes (VRAM for offloaded layers), regardless of where the
            // source tensor sits (lazily-read sources are always host-mapped)
            ggml_backend_buffer_type_t layer_buft = model.select_buft((int) il);
            if (strstr(ggml_backend_buft_name(layer_buft), "REPACK") != nullptr) {
                throw std::runtime_error("moe pool: layer buffer type is a REPACK type; rerun with --no-repack");
            }
            buft = layer_buft;
            ggml_context * ctx_ = ctx_for(layer_buft);
            ggml_tensor * pt = ggml_new_tensor_3d(ctx_, src->type, src->ne[0], src->ne[1], n_slots);
            ggml_format_name(pt, "moe_pool.blk.%u.%s", il, k.name);
            pool_tensors.push_back(pt);
            sources.push_back(src);
        }
        // descriptors
        for (uint32_t e = 0; e < n_expert; ++e) {
            moe::ExpertDescriptor d;
            d.key = { il, e };
            for (ggml_tensor * src : sources) {
                moe::TensorSlice sl;
                sl.name   = ggml_get_name(src);
                sl.source = src;
                sl.bytes  = src->nb[2];
                sl.offset = (uint64_t) e * src->nb[2];
                sl.type   = src->type;
                if (meta && store_mode != 3) {
                    const int64_t ti = gguf_find_tensor(meta, sl.name.c_str());
                    if (ti < 0) {
                        throw std::runtime_error("moe pool: tensor not found in GGUF: " + sl.name);
                    }
                    sl.file_offset = meta_data_offset + gguf_get_tensor_offset(meta, ti) + sl.offset;
                }
                d.total_bytes += sl.bytes;
                d.slices.push_back(std::move(sl));
            }
            index_.descs[ls * n_expert + e] = std::move(d);
        }
        pools_.push_back(std::make_unique<moe::LayerPool>(il, (uint32_t) n_slots, pool_tensors));
        // swap the graph-visible tensors for the pool tensors
        size_t pi = 0;
        for (const auto & k : kinds) {
            if (L.*k.field) {
                L.*k.field = pool_tensors[pi++];
            }
        }
    }

    size_t pool_bytes = 0;
    bool   all_host   = true;
    std::string buft_names;
    for (auto & [bt, c] : ctx_by_buft_) {
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, bt);
        if (!b) {
            throw std::runtime_error(std::string("moe pool: failed to allocate pool buffer in ") + ggml_backend_buft_name(bt));
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs_.push_back(b);
        pool_bytes += ggml_backend_buffer_get_size(b);
        all_host = all_host && ggml_backend_buffer_is_host(b);
        buft_names += (buft_names.empty() ? "" : "+") + std::string(ggml_backend_buft_name(bt));
    }

    if (meta) {
        gguf_free(meta);
    }

    // moepack layout (tools/moe-pack/moe_pack.py): 4 KiB header, then for each MoE layer in ascending order,
    // for each expert, the slices in kind order (up, down, gate as present), each bundle padded to `align`.
    // The layout is deterministic, so the offsets are computed here and checked against the file size; the
    // .moeidx JSON with per-slice hashes is the verification artifact on the Python side.

    switch (store_mode) {
        case 0:  store_ = std::make_unique<moe::MemoryExpertStore>();             break;
        case 1:  store_ = std::make_unique<moe::DirectIOExpertStore>(model_path); break;
        case 2:  store_ = std::make_unique<moe::IoUringExpertStore>(model_path);  break;
        default: throw std::runtime_error("moe pool: unknown store mode " + std::to_string(store_mode));
    }
    if (host_cache_bytes > 0 && store_mode >= 1) {
        // on a device pool, allocate the arena from the device's pinned host buffer type (DMA-speed H2D)
        ggml_backend_buffer_type_t host_buft = nullptr;
        if (!all_host) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        }
        // quant mixes give layers different expert sizes, so a slab must hold the largest one
        uint64_t max_bundle = 0;
        for (const auto & d : index_.descs) {
            max_bundle = std::max(max_bundle, d.total_bytes);
        }
        auto cache = std::make_unique<moe::CachingExpertStore>(std::move(store_), host_cache_bytes, max_bundle, host_buft);
        host_cache_ = cache.get();
        store_ = std::move(cache);
    }
    if (verify) {
        if (store_mode == 0) {
            throw std::runtime_error("moe pool: --moe-verify needs a non-reference store (use --moe-store file, direct or pack)");
        }
        store_ = std::make_unique<moe::VerifyingExpertStore>(std::move(store_), std::make_unique<moe::MemoryExpertStore>());
    }

    // Parallel fetch writes into disjoint byte ranges of the pool tensors from several threads, which is
    // safe for host buffers (a memcpy each) but not for device buffers, where the transfer goes through a
    // stream. Refuse rather than corrupt; the device path gets its own transfer scheduler in M6/M8.
    // Device pools read in parallel too; only the H2D copies are serialised on the calling thread.
    const int32_t n_io = io_threads > 0 ? io_threads : 1;
    for (auto & pool : pools_) {
        pool->set_host_pool(all_host);
        pool->set_io_threads((uint32_t) n_io);
    }

    if (!all_host) {
        device_ = ggml_backend_buft_get_device(buft);
        if (device_) {
            transfer_backend_ = ggml_backend_dev_init(device_, nullptr);
        }
    }

    if (prefetch_k > 0) {
        init_prerouter(model, prefetch_k);
        if (prerouter_.enabled) {
            model.hparams.moe_prefetch_k = prefetch_k;
            model.hparams.moe_prefetch_lookahead = prerouter_.lookahead;
        }
    }

    const auto & d0 = index_.descs[0];
    fprintf(stderr, "moe pool: per-layer pool, %zu MoE layers, %d slots per layer, %zu tensors/expert, %.2f MiB/expert, "
                    "pool buffer %.2f MiB (%s), store = %s, io threads = %d\n",
                    index_.moe_layers.size(), n_slots,
                    d0.slices.size(), d0.total_bytes / (1024.0 * 1024.0),
                    pool_bytes / (1024.0 * 1024.0), buft_names.c_str(), store_->name(), n_io);

    model.hparams.moe_pool_active = true;
}

void llama_moe_pool::attach_compute_backend(ggml_backend_t compute) {
    if (!async_enabled_ || !transfer_backend_ || !compute) {
        return;
    }
    for (auto & pool : pools_) {
        pool->set_device_backends(transfer_backend_, compute, device_);
    }
    fprintf(stderr, "moe pool: async H2D on a dedicated transfer stream (%s)\n", ggml_backend_name(transfer_backend_));
}

llama_moe_pool::~llama_moe_pool() {
    {
        std::lock_guard<std::mutex> lock(prerouter_.mtx);
        prerouter_.stop = true;
    }
    prerouter_.cv.notify_all();
    for (auto & th : prerouter_.threads) {
        th.join();
    }
    for (auto & pool : pools_) {
        pool->drain_transfers();
    }
    fprintf(stderr, "moe pool stats: %s\n", stats_json().c_str());
    for (auto * b : bufs_) {
        ggml_backend_buffer_free(b);
    }
    if (transfer_backend_) {
        ggml_backend_free(transfer_backend_);
    }
    for (auto & [bt, c] : ctx_by_buft_) {
        (void) bt;
        ggml_free(c);
    }
}

std::string llama_moe_pool::stats_json() const {
    std::string out = stats_.json();
    if (host_cache_) {
        const auto hs = host_cache_->stats();
        char buf[320];
        snprintf(buf, sizeof(buf),
                 " host_cache: {\"capacity\": %llu, \"resident\": %llu, \"hits\": %llu, \"misses\": %llu, "
                 "\"hit_rate\": %.4f, \"bytes_from_cache\": %llu, \"evictions\": %llu}",
                 (unsigned long long) host_cache_->capacity(), (unsigned long long) host_cache_->resident_bytes(),
                 (unsigned long long) hs.hits, (unsigned long long) hs.misses,
                 (hs.hits + hs.misses) ? (double) hs.hits / (double) (hs.hits + hs.misses) : 0.0,
                 (unsigned long long) hs.bytes_from_cache, (unsigned long long) hs.evictions);
        out += buf;
        if (prerouter_.enabled) {
            char pb[320];
            snprintf(pb, sizeof(pb),
                     " prerouter: {\"predictions\": %llu, \"issued\": %llu, \"prefetched\": %llu, \"useful\": %llu, "
                     "\"wasted\": %llu, \"dropped\": %llu, \"bytes_prefetched\": %llu}",
                     (unsigned long long) prerouter_.predictions, (unsigned long long) prerouter_.issued,
                     (unsigned long long) hs.prefetched, (unsigned long long) hs.prefetch_useful,
                     (unsigned long long) hs.prefetch_wasted, (unsigned long long) prerouter_.dropped,
                     (unsigned long long) hs.bytes_prefetched);
            out += pb;
        }
    }
    return out;
}

static int parse_slot_ids_layer(const char * name) {
    const size_t n = strlen(SLOT_IDS_PREFIX);
    if (strncmp(name, SLOT_IDS_PREFIX, n) != 0 || name[n] == '\0') {
        return -1;
    }
    for (const char * q = name + n; *q; ++q) {
        if (*q < '0' || *q > '9') {
            return -1;
        }
    }
    return atoi(name + n);
}

static int parse_prefixed_layer(const char * name, const std::string & prefix) {
    if (prefix.empty() || strncmp(name, prefix.c_str(), prefix.size()) != 0 || name[prefix.size()] == '\0') {
        return -1;
    }
    for (const char * q = name + prefix.size(); *q; ++q) {
        if (*q < '0' || *q > '9') {
            return -1;
        }
    }
    return atoi(name + prefix.size());
}

bool llama_moe_pool::cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (llama_moe_pool *) user_data;
    const int il = parse_slot_ids_layer(t->name);
    const bool ig = self->prerouter_.enabled;
    const int pf  = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_IDS_PREFIX)) : -1;
    const int pe  = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_ENTRY_IDS_PREFIX)) : -1;

    if (ask) {
        bool need = il >= 0 || pf >= 0 || pe >= 0;
        if (self->user_cb) {
            need = self->user_cb(t, true, self->user_ud) || need;
        }
        return need;
    }

    if (il >= 0) {
        self->on_slot_ids(t, (uint32_t) il);
    }
    if (pf >= 0) {
        self->on_prefetch_ids(t, (uint32_t) pf, false);
    }
    if (pe >= 0) {   // entry prediction: the named layer IS the target
        self->on_prefetch_ids(t, (uint32_t) pe, true);
    }
    if (self->user_cb) {
        return self->user_cb(t, false, self->user_ud);
    }
    return true;
}

void llama_moe_pool::init_prerouter(llama_model & model, int k) {
    (void) model;
    prerouter_.k = k;
    prerouter_.enabled = host_cache_ != nullptr;
    if (prerouter_.enabled) {
        for (int i = 0; i < 2; ++i) {
            prerouter_.threads.emplace_back([this]() { prefetch_loop(); });
        }
    }
    fprintf(stderr, "moe prerouter: %s (k=%d, lookahead=%d MoE layer(s), in-graph router nodes)\n",
            prerouter_.enabled ? "on" : "OFF - needs --moe-host-cache", prerouter_.k, prerouter_.lookahead);
}

void llama_moe_pool::prefetch_loop() {
    for (;;) {
        std::pair<uint32_t, uint32_t> item;
        {
            std::unique_lock<std::mutex> lock(prerouter_.mtx);
            prerouter_.cv.wait(lock, [&] { return prerouter_.stop || !prerouter_.queue.empty(); });
            if (prerouter_.stop) {
                return;
            }
            item = prerouter_.queue.back();   // newest first: the most recent prediction is the most urgent
            prerouter_.queue.pop_back();
        }
        try {
            host_cache_->prefetch(index_.get(moe::ExpertKey{ item.first, item.second }));
        } catch (...) {
            // a failed prefetch is not an error for the model: the demand path will read it itself
        }
    }
}

void llama_moe_pool::issue_prefetch(uint32_t il_t, const std::vector<int32_t> & ids, int64_t n_tok) {
    const uint32_t n_used = std::max<uint32_t>(1, index_.n_expert ? (uint32_t) (ids.size() / std::max<int64_t>(1, n_tok)) : 1);
    (void) n_used;
    std::vector<uint32_t> wanted;
    wanted.reserve(ids.size());
    const size_t k = n_tok > 0 ? ids.size() / (size_t) n_tok : ids.size();
    for (int64_t tk = 0; tk < n_tok; ++tk) {
        for (size_t j = 0; j < k; ++j) {
            const int32_t e = ids[(size_t) tk * k + j];
            if (e < 0 || (uint32_t) e >= index_.n_expert) {
                continue;
            }
            wanted.push_back((uint32_t) e);
        }
    }
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    prerouter_.predictions += wanted.size();

    auto & pool = pools_[(size_t) index_.layer_slot(il_t)];
    std::lock_guard<std::mutex> lock(prerouter_.mtx);
    const size_t cap = 4 * (size_t) prerouter_.k * (size_t) std::max<int64_t>(1, n_tok);
    if (prerouter_.queue.size() > cap) {
        prerouter_.dropped += prerouter_.queue.size();
        prerouter_.queue.clear();
    }
    for (uint32_t e : wanted) {
        moe::ExpertKey key{ il_t, e };
        if (pool->contains(key) || host_cache_->contains(key)) {
            continue;
        }
        prerouter_.queue.emplace_back(il_t, e);
        prerouter_.issued++;
    }
    prerouter_.cv.notify_all();
}

// In-graph prerouter output for MoE layer `layer`: I32 [k, n_tokens]. For the regular tensor the target is
// `lookahead` MoE layers ahead; for the entry tensor the named layer is itself the target.
void llama_moe_pool::on_prefetch_ids(struct ggml_tensor * t, uint32_t layer, bool entry) {
    if (!prerouter_.enabled || t->type != GGML_TYPE_I32) {
        return;
    }
    const int ls = index_.layer_slot(layer);
    if (ls < 0) {
        return;
    }
    uint32_t il_t = layer;
    if (!entry) {
        const size_t target = (size_t) ls + (size_t) prerouter_.lookahead;
        if (target >= index_.moe_layers.size()) {
            return;
        }
        il_t = index_.moe_layers[target];
    }
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<int32_t> ids(n);
    ggml_backend_tensor_get(t, ids.data(), 0, n * sizeof(int32_t));
    issue_prefetch(il_t, ids, t->ne[1]);
}

void llama_moe_pool::on_slot_ids(struct ggml_tensor * t, uint32_t il) {
    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const int ls = index_.layer_slot(il);
    if (ls < 0) {
        throw std::runtime_error("moe pool: slot-id tensor for a layer without a pool: " + std::to_string(il));
    }
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    const size_t  n      = (size_t) n_used * n_tok;

    ids_buf_.resize(n);
    // the tensor may be non-contiguous in theory; ggml_dup produces a contiguous result
    GGML_ASSERT(ggml_is_contiguous(t));
    ggml_backend_tensor_get(t, ids_buf_.data(), 0, n * sizeof(int32_t));

    keys_buf_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const int32_t e = ids_buf_[i];
        if (e < 0 || (uint32_t) e >= index_.n_expert) {
            throw std::runtime_error("moe pool: routed expert id out of range: " + std::to_string(e));
        }
        keys_buf_[i] = { il, (uint32_t) e };
    }
    // The scheduler synchronised the compute backend before this callback, so every previously issued
    // transfer (in any layer's pool) has completed: release their pinned source slabs now, globally.
    // Holding them until each pool's own next batch could pin the whole host cache and deadlock.
    for (auto & pool : pools_) {
        pool->drain_transfers();
    }
    pools_[(size_t) ls]->ensure(index_, *store_, keys_buf_, slots_buf_, stats_, clock_);
    ggml_backend_tensor_set(t, slots_buf_.data(), 0, n * sizeof(int32_t));
}
