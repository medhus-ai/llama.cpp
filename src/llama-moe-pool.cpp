#include "llama-moe-pool.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

static const char * SLOT_IDS_PREFIX = "ffn_moe_slot_ids-";

llama_moe_pool::llama_moe_pool(llama_model & model, int32_t n_slots_) : n_slots(n_slots_) {
    if (n_slots <= 0) {
        throw std::runtime_error("moe pool: n_slots must be > 0");
    }
    const uint32_t n_expert = model.hparams.n_expert;
    if (n_expert == 0) {
        throw std::runtime_error("moe pool: model has no routed experts");
    }
    if ((uint32_t) n_slots > n_expert) {
        LLAMA_LOG_WARN("%s: n_slots (%d) > n_expert (%u), clamping\n", __func__, n_slots, n_expert);
        n_slots = (int32_t) n_expert;
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
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors, nullptr, true };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        throw std::runtime_error("moe pool: ggml_init failed");
    }

    ggml_backend_buffer_type_t buft = nullptr;
    index_.descs.resize(index_.moe_layers.size() * n_expert);

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
            ggml_backend_buffer_type_t src_buft = ggml_backend_buffer_get_type(src->buffer);
            const char * src_buft_name = ggml_backend_buft_name(src_buft);
            if (strstr(src_buft_name, "REPACK") != nullptr) {
                throw std::runtime_error(std::string("moe pool: expert tensors are in the ") + src_buft_name +
                                         " buffer type, whose contents cannot be read back; rerun with --no-repack");
            }
            if (!buft) {
                buft = src_buft;
            } else if (buft != src_buft) {
                throw std::runtime_error("moe pool: expert tensors live in different buffer types; not supported yet");
            }
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

    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);
    if (!buf_) {
        throw std::runtime_error("moe pool: failed to allocate pool buffer");
    }
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    store_ = std::make_unique<moe::MemoryExpertStore>();

    const auto & d0 = index_.descs[0];
    fprintf(stderr, "moe pool: compact expert pool: %zu MoE layers x %d slots, %zu tensors/expert, %.2f MiB/expert, "
                   "pool buffer %.2f MiB (%s), store = %s\n",
                   index_.moe_layers.size(), n_slots, d0.slices.size(), d0.total_bytes / (1024.0 * 1024.0),
                   ggml_backend_buffer_get_size(buf_) / (1024.0 * 1024.0), ggml_backend_buft_name(buft), store_->name());

    model.hparams.moe_pool_active = true;
}

llama_moe_pool::~llama_moe_pool() {
    fprintf(stderr, "moe pool stats: %s\n", stats_json().c_str());
    if (buf_) {
        ggml_backend_buffer_free(buf_);
    }
    if (ctx_) {
        ggml_free(ctx_);
    }
}

std::string llama_moe_pool::stats_json() const {
    return stats_.json();
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

bool llama_moe_pool::cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * self = (llama_moe_pool *) user_data;
    const int il = parse_slot_ids_layer(t->name);

    if (ask) {
        bool need = il >= 0;
        if (self->user_cb) {
            need = self->user_cb(t, true, self->user_ud) || need;
        }
        return need;
    }

    if (il >= 0) {
        self->on_slot_ids(t, (uint32_t) il);
    }
    if (self->user_cb) {
        return self->user_cb(t, false, self->user_ud);
    }
    return true;
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
    pools_[ls]->ensure(index_, *store_, keys_buf_, slots_buf_, stats_, clock_);
    ggml_backend_tensor_set(t, slots_buf_.data(), 0, n * sizeof(int32_t));
}
