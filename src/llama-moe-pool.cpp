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
static const char * PREFETCH_PROBS_PREFIX = "ffn_moe_prefetch_probs-";
static const char * PREFETCH_ENTRY_IDS_PREFIX = "ffn_moe_prefetch_entry_ids-";
static const char * PREFETCH_ENTRY_PROBS_PREFIX = "ffn_moe_prefetch_entry_probs-";

llama_moe_pool::llama_moe_pool(llama_model & model, int32_t n_slots_, int32_t store_mode, const std::string & model_path, bool verify, bool shared, int32_t io_threads, const std::string & pack_path, uint64_t host_cache_bytes,
                               int32_t prefetch_k, int32_t prefetch_lookahead, const char * prefetch_src)
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

    // A single pool shared by every MoE layer needs all of them to have identically shaped and typed
    // expert tensors. Check that before committing to the shared layout.
    bool uniform = true;
    {
        const uint32_t il0 = index_.moe_layers[0];
        for (uint32_t il : index_.moe_layers) {
            for (const auto & k : kinds) {
                const ggml_tensor * a_t = model.layers[il0].*k.field;
                const ggml_tensor * b_t = model.layers[il].*k.field;
                if ((a_t == nullptr) != (b_t == nullptr)) {
                    uniform = false;
                } else if (a_t && (a_t->type != b_t->type || a_t->ne[0] != b_t->ne[0] ||
                                   a_t->ne[1] != b_t->ne[1] || a_t->ne[2] != b_t->ne[2])) {
                    uniform = false;
                }
            }
        }
    }
    if (shared && !uniform) {
        fprintf(stderr, "moe pool: expert tensors differ between MoE layers, falling back to per-layer pools\n");
        shared = false;
    }
    shared_ = shared;

    // Cap the pool at the number of distinct experts it could ever hold: n_expert per layer, or
    // n_expert * n_moe_layers when one pool serves every layer.
    {
        const uint64_t cap = shared_ ? (uint64_t) n_expert * index_.moe_layers.size() : n_expert;
        if ((uint64_t) n_slots > cap) {
            fprintf(stderr, "moe pool: %d slots exceeds the %llu distinct experts a %s pool can hold, clamping\n",
                    n_slots, (unsigned long long) cap, shared_ ? "shared" : "per-layer");
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
            if (shared_ && buft && buft != layer_buft) {
                throw std::runtime_error("moe pool: --moe-pool-shared needs every MoE layer on the same device");
            }
            buft = layer_buft;
            ggml_context * ctx_ = ctx_for(layer_buft);
            ggml_tensor * pt = nullptr;
            if (shared_) {
                auto it = shared_tensors_.find(k.name);
                if (it == shared_tensors_.end()) {
                    pt = ggml_new_tensor_3d(ctx_, src->type, src->ne[0], src->ne[1], n_slots);
                    ggml_format_name(pt, "moe_pool.shared.%s", k.name);
                    shared_tensors_[k.name] = pt;
                } else {
                    pt = it->second;
                }
            } else {
                pt = ggml_new_tensor_3d(ctx_, src->type, src->ne[0], src->ne[1], n_slots);
                ggml_format_name(pt, "moe_pool.blk.%u.%s", il, k.name);
            }
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
        if (shared_) {
            if (pools_.empty()) {
                pools_.push_back(std::make_unique<moe::LayerPool>(moe::LayerPool::ANY_LAYER, (uint32_t) n_slots, pool_tensors));
            }
        } else {
            pools_.push_back(std::make_unique<moe::LayerPool>(il, (uint32_t) n_slots, pool_tensors));
        }
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
    if (store_mode == 3) {
        const uint64_t align = 4096;
        struct stat st;
        if (stat(pack_path.c_str(), &st) != 0) {
            throw std::runtime_error("moe pool: cannot stat moepack " + pack_path + " (run tools/moe-pack/moe_pack.py)");
        }
        const uint64_t bundle_payload = index_.descs[0].total_bytes;
        const uint64_t bundle_len = ((bundle_payload + align - 1) / align) * align;
        const uint64_t expected = align + bundle_len * (uint64_t) index_.descs.size();
        if ((uint64_t) st.st_size != expected) {
            throw std::runtime_error("moe pool: moepack size " + std::to_string(st.st_size) + " != expected " +
                                     std::to_string(expected) + " for this model's expert layout; re-pack it");
        }
        for (size_t i = 0; i < index_.descs.size(); ++i) {
            uint64_t off = align + bundle_len * i;
            for (auto & sl : index_.descs[i].slices) {
                sl.file_offset = off;
                off += sl.bytes;
            }
        }
    }

    switch (store_mode) {
        case 0:  store_ = std::make_unique<moe::MemoryExpertStore>();               break;
        case 1:  store_ = std::make_unique<moe::BufferedFileExpertStore>(model_path); break;
        case 2:  store_ = std::make_unique<moe::DirectIOExpertStore>(model_path);   break;
        case 3:  store_ = std::make_unique<moe::PackExpertStore>(pack_path);        break;
        case 4:  store_ = std::make_unique<moe::IoUringExpertStore>(model_path);    break;
        default: throw std::runtime_error("moe pool: unknown store mode " + std::to_string(store_mode));
    }
    if (host_cache_bytes > 0 && store_mode >= 1) {
        // on a device pool, allocate the arena from the device's pinned host buffer type (DMA-speed H2D)
        ggml_backend_buffer_type_t host_buft = nullptr;
        if (!all_host) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        }
        auto cache = std::make_unique<moe::CachingExpertStore>(std::move(store_), host_cache_bytes, index_.descs[0].total_bytes, host_buft);
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
        prerouter_in_graph_ = (prefetch_src == nullptr);
        init_prerouter(model, prefetch_k, prefetch_lookahead, prefetch_src ? prefetch_src : "attn_norm");
        if (prerouter_.enabled && prerouter_in_graph_) {
            model.hparams.moe_prefetch_k = prefetch_k;
            model.hparams.moe_prefetch_lookahead = prefetch_lookahead < 1 ? 1 : prefetch_lookahead;
        }
    }

    const auto & d0 = index_.descs[0];
    fprintf(stderr, "moe pool: %s pool, %zu MoE layers, %d slots%s, %zu tensors/expert, %.2f MiB/expert, "
                    "pool buffer %.2f MiB (%s), store = %s, io threads = %d\n",
                    shared_ ? "shared" : "per-layer", index_.moe_layers.size(), n_slots,
                    shared_ ? " total" : " per layer", d0.slices.size(), d0.total_bytes / (1024.0 * 1024.0),
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
                     (unsigned long long) (hs.prefetched * index_.descs[0].total_bytes));
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
    int blk = (self->prerouter_.enabled && !self->prerouter_in_graph_) ? parse_prefixed_layer(t->name, self->prerouter_.src_prefix) : -1;
    if (blk >= 0 && self->index_.layer_slot((uint32_t) blk) < 0) {
        blk = -1;   // only the normed input of MoE layers; those splits already exist for the slot-id tensor
    }
    const bool ig = self->prerouter_.enabled && self->prerouter_in_graph_;
    const int pf  = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_IDS_PREFIX)) : -1;
    const int pp  = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_PROBS_PREFIX)) : -1;
    const int pe  = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_ENTRY_IDS_PREFIX)) : -1;
    const int pep = ig ? parse_prefixed_layer(t->name, std::string(PREFETCH_ENTRY_PROBS_PREFIX)) : -1;

    if (ask) {
        bool need = il >= 0 || blk >= 0 || pf >= 0 || pp >= 0 || pe >= 0 || pep >= 0;
        if (self->user_cb) {
            need = self->user_cb(t, true, self->user_ud) || need;
        }
        return need;
    }

    if (il >= 0) {
        self->on_slot_ids(t, (uint32_t) il);
    }
    if (blk >= 0) {
        self->on_residual(t, (uint32_t) blk);
    }
    if (pf >= 0) {
        self->on_prefetch_ids(t, (uint32_t) pf);
    }
    if (pe >= 0) {   // entry prediction: the named layer IS the target
        self->pending_entry_ = true;
        self->on_prefetch_ids(t, (uint32_t) pe);
    }
    if (pp >= 0) {
        self->on_prefetch_probs(t, (uint32_t) pp, false);
    }
    if (pep >= 0) {
        self->on_prefetch_probs(t, (uint32_t) pep, true);
    }
    if (self->user_cb) {
        return self->user_cb(t, false, self->user_ud);
    }
    return true;
}

void llama_moe_pool::init_prerouter(llama_model & model, int k, int lookahead, const std::string & src_prefix) {
    prerouter_.k = k;
    prerouter_.lookahead = lookahead < 1 ? 1 : lookahead;
    prerouter_.src_prefix = src_prefix + "-";
    prerouter_.layers.clear();
    for (uint32_t il : index_.moe_layers) {
        auto & L = model.layers[il];
        if (!L.attn_norm || !L.ffn_gate_inp) {
            throw std::runtime_error("moe prerouter: layer " + std::to_string(il) + " lacks attn_norm/ffn_gate_inp");
        }
        prerouter_t::layer_t ly;
        ly.il = il;
        ly.n_embd = (uint32_t) L.ffn_gate_inp->ne[0];
        ly.n_expert = (uint32_t) L.ffn_gate_inp->ne[1];
        auto pull = [](ggml_tensor * t, std::vector<float> & out) {
            if (t->type != GGML_TYPE_F32) {
                throw std::runtime_error(std::string("moe prerouter: expected F32 tensor ") + ggml_get_name(t));
            }
            out.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        };
        pull(L.attn_norm, ly.norm);
        pull(L.ffn_gate_inp, ly.gate);
        if (L.ffn_exp_probs_b) {
            pull(L.ffn_exp_probs_b, ly.bias);
        } else {
            ly.bias.assign(ly.n_expert, 0.0f);
        }
        prerouter_.layers.push_back(std::move(ly));
    }
    prerouter_.enabled = host_cache_ != nullptr;
    if (prerouter_.enabled) {
        for (int i = 0; i < 2; ++i) {
            prerouter_.threads.emplace_back([this]() { prefetch_loop(); });
        }
    }
    fprintf(stderr, "moe prerouter: %s (k=%d, lookahead=%d MoE layer(s), %s)\n",
            prerouter_.enabled ? "on" : "OFF - needs --moe-host-cache", prerouter_.k, prerouter_.lookahead,
            prerouter_in_graph_ ? "in-graph router nodes" : ("CPU router on " + prerouter_.src_prefix + "<moe layer>").c_str());
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

// The normed input of MoE layer `layer` (attn_norm-<layer>, already a callback split point) is available:
// run the router of the MoE layer `lookahead` layers ahead on it and queue its non-resident top-k experts.
// attn_norm-<layer> = rmsnorm(x) * g_layer; the target router wants rmsnorm(x) * g_target.
void llama_moe_pool::on_residual(struct ggml_tensor * t, uint32_t layer) {
    if (!prerouter_.enabled || t->type != GGML_TYPE_F32) {
        return;
    }
    const int ls = index_.layer_slot(layer);
    if (ls < 0) {
        return;
    }
    const size_t target = (size_t) ls + (size_t) prerouter_.lookahead;
    if (target >= index_.moe_layers.size()) {
        return;
    }
    const auto & src = prerouter_.layers[(size_t) ls];
    const auto & ly  = prerouter_.layers[target];
    const int64_t n_embd = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    if ((uint32_t) n_embd != ly.n_embd || !ggml_is_contiguous(t)) {
        return;
    }
    prerouter_.x.resize((size_t) n_embd * n_tok);
    ggml_backend_tensor_get(t, prerouter_.x.data(), 0, prerouter_.x.size() * sizeof(float));

    const uint32_t E = ly.n_expert;
    std::vector<float> xn((size_t) n_embd);
    std::vector<float> lg(E);
    std::vector<uint32_t> idx(E);
    std::vector<uint32_t> wanted;
    for (int64_t tk = 0; tk < n_tok; ++tk) {
        const float * xr = prerouter_.x.data() + tk * n_embd;
        for (int64_t i = 0; i < n_embd; ++i) {
            const float g = src.norm[i];
            xn[i] = (fabsf(g) > 1e-8f ? xr[i] / g : 0.0f) * ly.norm[i];
        }
        for (uint32_t e = 0; e < E; ++e) {
            const float * w = ly.gate.data() + (size_t) e * n_embd;
            float acc = 0.0f;
            for (int64_t i = 0; i < n_embd; ++i) { acc += w[i] * xn[i]; }
            lg[e] = 1.0f / (1.0f + expf(-acc)) + ly.bias[e];
        }
        for (uint32_t e = 0; e < E; ++e) { idx[e] = e; }
        const size_t kk = std::min<size_t>((size_t) prerouter_.k, E);
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](uint32_t a, uint32_t b) { return lg[a] > lg[b]; });
        wanted.insert(wanted.end(), idx.begin(), idx.begin() + kk);
    }
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    prerouter_.predictions += wanted.size();

    const int tls = index_.layer_slot(ly.il);
    auto & pool = pools_[shared_ ? 0 : (size_t) tls];
    std::lock_guard<std::mutex> lock(prerouter_.mtx);
    // a fresh prediction supersedes stale queued work: drop anything older than one layer's worth
    const size_t cap = 4 * (size_t) prerouter_.k * (size_t) std::max<int64_t>(1, n_tok);
    if (prerouter_.queue.size() > cap) {
        prerouter_.dropped += prerouter_.queue.size();
        prerouter_.queue.clear();
    }
    for (uint32_t e : wanted) {
        moe::ExpertKey key{ ly.il, e };
        if (pool->contains(key) || host_cache_->contains(key)) {
            continue;
        }
        prerouter_.queue.emplace_back(ly.il, e);
        prerouter_.issued++;
    }
    prerouter_.cv.notify_all();
}

void llama_moe_pool::issue_prefetch(uint32_t il_t, const std::vector<int32_t> & ids, const std::vector<float> * probs, int64_t n_tok) {
    const uint32_t n_used = std::max<uint32_t>(1, index_.n_expert ? (uint32_t) (ids.size() / std::max<int64_t>(1, n_tok)) : 1);
    (void) n_used;
    std::vector<uint32_t> wanted;
    wanted.reserve(ids.size());
    const size_t k = n_tok > 0 ? ids.size() / (size_t) n_tok : ids.size();
    for (int64_t tk = 0; tk < n_tok; ++tk) {
        // candidates are sorted by probability within a token (argsort_top_k); the margin keeps those whose
        // probability is within (1 - margin) of the n_expert_used-th best - the ones that could plausibly
        // make the real top-6 - and drops the tail that only pads the queue
        float thresh = -1.0f;
        if (probs && prefetch_margin_ > 0.0f) {
            const size_t ref = std::min<size_t>(k, 6) - 1;
            thresh = (*probs)[(size_t) tk * k + ref] * (1.0f - prefetch_margin_);
        }
        for (size_t j = 0; j < k; ++j) {
            const int32_t e = ids[(size_t) tk * k + j];
            if (e < 0 || (uint32_t) e >= index_.n_expert) {
                continue;
            }
            if (probs && prefetch_margin_ > 0.0f && (*probs)[(size_t) tk * k + j] < thresh) {
                break;   // sorted: everything after is lower
            }
            wanted.push_back((uint32_t) e);
        }
    }
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    prerouter_.predictions += wanted.size();

    auto & pool = pools_[shared_ ? 0 : (size_t) index_.layer_slot(il_t)];
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
// `lookahead` MoE layers ahead; for the entry tensor (pending_entry_) the named layer is the target.
void llama_moe_pool::on_prefetch_ids(struct ggml_tensor * t, uint32_t layer) {
    if (!prerouter_.enabled || t->type != GGML_TYPE_I32) {
        pending_entry_ = false;
        return;
    }
    const int ls = index_.layer_slot(layer);
    if (ls < 0) {
        pending_entry_ = false;
        return;
    }
    uint32_t il_t;
    if (pending_entry_) {
        il_t = layer;
    } else {
        const size_t target = (size_t) ls + (size_t) prerouter_.lookahead;
        if (target >= index_.moe_layers.size()) {
            return;
        }
        il_t = index_.moe_layers[target];
    }
    const size_t n = (size_t) ggml_nelements(t);
    pending_ids_.resize(n);
    ggml_backend_tensor_get(t, pending_ids_.data(), 0, n * sizeof(int32_t));
    pending_layer_ = il_t;
    pending_ntok_  = t->ne[1];
    if (prefetch_margin_ <= 0.0f) {
        issue_prefetch(il_t, pending_ids_, nullptr, pending_ntok_);
        pending_layer_ = UINT32_MAX;
        pending_entry_ = false;
    }
    // else: wait for the probs tensor, which the graph computes right after the ids
}

void llama_moe_pool::on_prefetch_probs(struct ggml_tensor * t, uint32_t layer, bool entry) {
    (void) layer; (void) entry;
    if (pending_layer_ == UINT32_MAX || t->type != GGML_TYPE_F32) {
        return;
    }
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float> probs(n);
    ggml_backend_tensor_get(t, probs.data(), 0, n * sizeof(float));
    if (n == pending_ids_.size()) {
        issue_prefetch(pending_layer_, pending_ids_, &probs, pending_ntok_);
    }
    pending_layer_ = UINT32_MAX;
    pending_entry_ = false;
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
    pools_[shared_ ? 0 : (size_t) ls]->ensure(index_, *store_, keys_buf_, slots_buf_, stats_, clock_);
    ggml_backend_tensor_set(t, slots_buf_.data(), 0, n * sizeof(int32_t));
}
