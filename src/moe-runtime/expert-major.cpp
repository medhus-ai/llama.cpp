#include "expert-major.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <algorithm>

namespace moe {

// slice names are the source tensor names, e.g. blk.3.ffn_down_exps.weight / .scale
static int find_slice(const ExpertDescriptor & e, const char * want, const char * suffix = ".weight") {
    for (size_t i = 0; i < e.slices.size(); ++i) {
        const std::string & n = e.slices[i].name;
        const size_t at = n.find(want);
        if (at != std::string::npos && n.compare(at + strlen(want), strlen(suffix), suffix) == 0) {
            return (int) i;
        }
    }
    return -1;
}

ExpertMajorFFN::ExpertMajorFFN(ggml_backend_t backend, const ExpertDescriptor & proto,
                               int64_t n_embd, int64_t n_ff, int64_t n_expert, int64_t n_expert_used, act a,
                               int64_t wave)
    : backend_(backend), wave_(std::max<int64_t>(1, std::min(wave, n_expert))),
      n_embd_(n_embd), n_ff_(n_ff), n_expert_(n_expert), n_used_(n_expert_used), act_(a) {
    slice_gate_ = find_slice(proto, "ffn_gate_exps");
    slice_up_   = find_slice(proto, "ffn_up_exps");
    slice_down_ = find_slice(proto, "ffn_down_exps");
    slice_gate_up_ = find_slice(proto, "ffn_gate_up_exps");
    slice_gate_s_  = find_slice(proto, "ffn_gate_exps", ".scale");
    slice_up_s_    = find_slice(proto, "ffn_up_exps",   ".scale");
    slice_down_s_  = find_slice(proto, "ffn_down_exps", ".scale");
    for (int sl : { slice_gate_s_, slice_up_s_, slice_down_s_ }) {
        if (sl >= 0 && (proto.slices[sl].bytes != sizeof(float) || proto.slices[sl].type != GGML_TYPE_F32)) {
            throw std::runtime_error("moe: expert-major expects f32 per-expert scales");
        }
    }
    if ((slice_up_ < 0 && slice_gate_up_ < 0) || slice_down_ < 0) {
        throw std::runtime_error("moe: expert-major needs a down slice and either up or gate_up");
    }
    build_index_buffers();
    slots_for(proto);
    by_expert_.resize((size_t) n_expert_);
}

void ExpertMajorFFN::build_index_buffers() {
    // Worst case every (rank, token) pair lands on one expert, so index buffers are sized for the
    // largest chunk we allow. 65536 pairs covers an 8192-token chunk at top-8.
    max_rows_ = 65536;

    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, /*no_alloc*/ true };
    ctx_  = ggml_init(ip);
    gidx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, max_rows_);
    sidx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I64, max_rows_);
    buf_  = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buf_) {
        throw std::runtime_error("moe: expert-major could not allocate its index buffers");
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    // one chain of up to ~14 nodes per expert in a wave (3 optional scales), plus slack
    graph_mem_.resize(ggml_tensor_overhead() * (size_t) (wave_ * 16 + 32) + ggml_graph_overhead_custom(wave_ * 16 + 32, false));
}

ExpertMajorFFN::slot_set & ExpertMajorFFN::slots_for(const ExpertDescriptor & d) {
    auto type_of = [&](int slice) { return slice >= 0 ? (int) d.slices[slice].type : (int) GGML_TYPE_COUNT; };
    const type_key key = { type_of(slice_gate_), type_of(slice_up_), type_of(slice_gate_up_), type_of(slice_down_) };
    auto it = sets_.find(key);
    if (it != sets_.end()) {
        return it->second;
    }

    slot_set & ss = sets_[key];
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (wave_ * 4 + 4), nullptr, /*no_alloc*/ true };
    ss.ctx = ggml_init(ip);

    ss.up.assign((size_t) wave_, nullptr);
    ss.down.resize((size_t) wave_);
    ss.gate.assign((size_t) wave_, nullptr);
    ss.gate_up.assign((size_t) wave_, nullptr);
    for (int64_t w = 0; w < wave_; ++w) {
        if (slice_up_ >= 0) {
            ss.up[(size_t) w] = ggml_new_tensor_2d(ss.ctx, d.slices[slice_up_].type, n_embd_, n_ff_);
        }
        ss.down[(size_t) w] = ggml_new_tensor_2d(ss.ctx, d.slices[slice_down_].type, n_ff_, n_embd_);
        if (slice_gate_up_ >= 0) {
            ss.gate_up[(size_t) w] = ggml_new_tensor_2d(ss.ctx, d.slices[slice_gate_up_].type, n_embd_, 2 * n_ff_);
        }
        if (slice_gate_ >= 0) {
            ss.gate[(size_t) w] = ggml_new_tensor_2d(ss.ctx, d.slices[slice_gate_].type, n_embd_, n_ff_);
        }
    }
    ss.buf = ggml_backend_alloc_ctx_tensors(ss.ctx, backend_);
    if (!ss.buf) {
        throw std::runtime_error("moe: expert-major could not allocate its expert slots");
    }

    size_t widest = d.slices[slice_down_].bytes;
    for (int sl : { slice_up_, slice_gate_up_, slice_gate_ }) {
        if (sl >= 0) {
            widest = std::max(widest, d.slices[sl].bytes);
        }
    }
    if (staging_.size() < widest) {
        staging_.resize(widest);
    }
    return ss;
}

float ExpertMajorFFN::read_scale(ExpertStore & store, const ExpertDescriptor & d, int slice) {
    if (slice < 0) {
        return 1.0f;
    }
    float v = 1.0f;
    store.read_slice(d, (size_t) slice, &v);
    bytes_fetched_ += sizeof(float);
    return v;
}

ExpertMajorFFN::~ExpertMajorFFN() {
    for (auto & [key, ss] : sets_) {
        if (ss.buf) ggml_backend_buffer_free(ss.buf);
        if (ss.ctx) ggml_free(ss.ctx);
    }
    if (galloc_) ggml_gallocr_free(galloc_);
    if (buf_)    ggml_backend_buffer_free(buf_);
    if (ctx_)    ggml_free(ctx_);
}

size_t ExpertMajorFFN::run(ExpertStore & store, const ExpertIndex & index, uint32_t layer,
                           const int32_t * ids, int64_t n_tokens, ggml_tensor * inp, ggml_tensor * out,
                           float swiglu_limit) {
    const int64_t n_pairs = n_tokens * n_used_;
    if (n_pairs > max_rows_) {
        throw std::runtime_error("moe: expert-major chunk exceeds its index buffers");
    }

    for (auto & v : by_expert_) {
        v.clear();
    }
    for (int64_t i = 0; i < n_pairs; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= (int32_t) n_expert_) {
            throw std::runtime_error("moe: expert-major saw an out-of-range expert id");
        }
        by_expert_[(size_t) e].push_back((int32_t) i);
    }

    size_t visited = 0;

    // Walk the experts that this chunk actually uses, `wave_` at a time. Each wave makes its experts
    // resident, then runs one graph holding every chain in the wave: one submission instead of one
    // per expert, which is what the per-expert version spent all its time on.
    std::vector<int64_t> used;
    used.reserve((size_t) n_expert_);
    for (int64_t e = 0; e < n_expert_; ++e) {
        if (!by_expert_[(size_t) e].empty()) {
            used.push_back(e);
        }
    }

    // the graph hands over [n_embd, 1, n_tokens] and [n_embd, n_used, n_ubatch]; index them as rows
    const int64_t inp_rows = inp->ne[1] * inp->ne[2] * inp->ne[3];
    const int64_t out_rows = out->ne[1] * out->ne[2] * out->ne[3];
    if (!ggml_is_contiguous(inp) || !ggml_is_contiguous(out)) {
        throw std::runtime_error("moe: expert-major needs contiguous input and output");
    }

    // every expert of a layer shares its slice types, so expert 0 picks the slot set
    slot_set & ss = slots_for(index.get(ExpertKey{ layer, 0 }));

    auto * cache = dynamic_cast<CachingExpertStore *>(&store);
    if (cache && !cache->pinned()) {
        cache = nullptr;   // an unpinned tier gains nothing from zero-copy; use the staging path
    }
    std::vector<std::thread> prefetchers;

    for (size_t base = 0; base < used.size(); base += (size_t) wave_) {
        const size_t n_in_wave = std::min((size_t) wave_, used.size() - base);

        // pack this wave's row lists back to back so one upload covers them all
        int64_t off = 0;
        std::vector<int64_t> starts(n_in_wave), counts(n_in_wave);
        rows32_.clear();
        rows64_.clear();
        for (size_t w = 0; w < n_in_wave; ++w) {
            const auto & rows = by_expert_[(size_t) used[base + w]];
            starts[w] = off;
            counts[w] = (int64_t) rows.size();
            for (int32_t p : rows) {
                rows32_.push_back(p / (int32_t) n_used_);   // gather by token
                rows64_.push_back((int64_t) p);             // scatter by (rank, token) pair
            }
            off += (int64_t) rows.size();
        }
        if ((int64_t) rows32_.size() > max_rows_) {
            throw std::runtime_error("moe: expert-major wave exceeds its index buffers");
        }
        for (size_t r = 0; r < rows32_.size(); ++r) {
            if (rows32_[r] < 0 || rows32_[r] >= (int32_t) inp_rows) {
                fprintf(stderr, "moe: expert-major gather row %d out of [0,%lld) (layer %u)\n",
                        rows32_[r], (long long) inp_rows, layer);
                throw std::runtime_error("moe: expert-major gather index out of range");
            }
            if (rows64_[r] < 0 || rows64_[r] >= out_rows) {
                fprintf(stderr, "moe: expert-major scatter row %lld out of [0,%lld) (layer %u)\n",
                        (long long) rows64_[r], (long long) out_rows, layer);
                throw std::runtime_error("moe: expert-major scatter index out of range");
            }
        }
        ggml_backend_tensor_set(gidx_, rows32_.data(), 0, rows32_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(sidx_, rows64_.data(), 0, rows64_.size() * sizeof(int64_t));

        // Prefetch the wave after this one into the host tier while this wave loads and computes:
        // the SSD reads then overlap the GEMMs instead of serialising in front of them.
        for (auto & t : prefetchers) { t.join(); }
        prefetchers.clear();
        if (cache) {
            const size_t nb = base + (size_t) wave_;
            const size_t ne = std::min(used.size(), nb + (size_t) wave_);
            if (nb < ne) {
                const int n_thr = 8;
                for (int t = 0; t < n_thr; ++t) {
                    prefetchers.emplace_back([&, t, nb, ne]() {
                        for (size_t i = nb + (size_t) t; i < ne; i += (size_t) n_thr) {
                            const ExpertDescriptor & d = index.get(ExpertKey{ layer, (uint32_t) used[i] });
                            if (cache->can_acquire(d)) { cache->prefetch(d); }
                        }
                    });
                }
            }
        }

        std::vector<float> s_gate(n_in_wave, 1.0f), s_up(n_in_wave, 1.0f), s_down(n_in_wave, 1.0f);
        std::vector<const ExpertDescriptor *> held;
        held.reserve(n_in_wave);
        for (size_t w = 0; w < n_in_wave; ++w) {
            const ExpertDescriptor & d = index.get(ExpertKey{ layer, (uint32_t) used[base + w] });
            auto check = [&](int slice, ggml_tensor * dst) {
                if (d.slices[slice].bytes > ggml_nbytes(dst)) {
                    throw std::runtime_error("moe: expert-major slot does not match this layer's expert slice");
                }
            };
            if (cache && cache->can_acquire(d)) {
                // zero-copy: the bundle sits pinned in the host tier, so the upload is one DMA per
                // slice on this backend's stream and the GEMMs queue behind it. Released after compute.
                const uint8_t * b = cache->acquire_bundle(d);
                held.push_back(&d);
                uint64_t off = 0;
                for (size_t k = 0; k < d.slices.size(); ++k) {
                    const uint64_t bytes = d.slices[k].bytes;
                    ggml_tensor * dst = nullptr;
                    if      ((int) k == slice_gate_up_) dst = ss.gate_up[w];
                    else if ((int) k == slice_gate_)    dst = ss.gate[w];
                    else if ((int) k == slice_up_)      dst = ss.up[w];
                    else if ((int) k == slice_down_)    dst = ss.down[w];
                    if (dst) {
                        check((int) k, dst);
                        ggml_backend_tensor_set_async(backend_, dst, b + off, 0, bytes);
                        bytes_fetched_ += bytes;
                    } else if ((int) k == slice_gate_s_) { memcpy(&s_gate[w], b + off, sizeof(float)); }
                    else if   ((int) k == slice_up_s_)   { memcpy(&s_up[w],   b + off, sizeof(float)); }
                    else if   ((int) k == slice_down_s_) { memcpy(&s_down[w], b + off, sizeof(float)); }
                    off += bytes;
                }
                continue;
            }
            s_gate[w] = read_scale(store, d, slice_gate_s_);
            s_up[w]   = read_scale(store, d, slice_up_s_);
            s_down[w] = read_scale(store, d, slice_down_s_);
            auto load = [&](int slice, ggml_tensor * dst) {
                if (slice < 0 || !dst) {
                    return;
                }
                const auto & sl = d.slices[slice];
                check(slice, dst);
                if (sl.bytes > staging_.size()) {
                    staging_.resize(sl.bytes);
                }
                store.read_slice(d, (size_t) slice, staging_.data());
                ggml_backend_tensor_set(dst, staging_.data(), 0, sl.bytes);
                bytes_fetched_ += sl.bytes;
            };
            load(slice_gate_up_, ss.gate_up[w]);
            load(slice_gate_, ss.gate[w]);
            load(slice_up_,   ss.up[w]);
            load(slice_down_, ss.down[w]);
        }

        ggml_init_params gp = { graph_mem_.size(), graph_mem_.data(), /*no_alloc*/ true };
        ggml_context * gc = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(gc, (size_t) (wave_ * 16 + 32), false);
        // Leaves that alias the caller's memory. `inp` is a node of the model graph: a view of it
        // would drag that whole graph in as parents when the wave graph is expanded.
        auto alias_2d = [&](ggml_tensor * t, int64_t rows) {
            ggml_tensor * a = ggml_new_tensor_2d(gc, t->type, t->ne[0], rows);
            a->data   = t->data;
            a->buffer = t->buffer;
            return a;
        };
        ggml_tensor * inp2 = alias_2d(inp, inp_rows);
        ggml_tensor * out2 = alias_2d(out, out_rows);

        for (size_t w = 0; w < n_in_wave; ++w) {
            const int64_t m = counts[w];
            ggml_tensor * gv = ggml_view_1d(gc, gidx_, m, (size_t) starts[w] * sizeof(int32_t));
            ggml_tensor * sv = ggml_view_1d(gc, sidx_, m, (size_t) starts[w] * sizeof(int64_t));
            ggml_tensor * x  = ggml_get_rows(gc, inp2, gv);
            ggml_tensor * h  = nullptr;
            if (ss.gate_up[w]) {
                // one GEMM, then split: gate = rows [0, n_ff), up = rows [n_ff, 2*n_ff)
                ggml_tensor * gu = ggml_mul_mat(gc, ss.gate_up[w], x);   // [2*n_ff, m]
                if (s_up[w] != 1.0f) {
                    gu = ggml_scale(gc, gu, s_up[w]);
                }
                ggml_tensor * g  = ggml_view_2d(gc, gu, n_ff_, gu->ne[1], gu->nb[1], 0);
                ggml_tensor * u  = ggml_view_2d(gc, gu, n_ff_, gu->ne[1], gu->nb[1], n_ff_ * gu->nb[0]);
                // same op as the token-major path (works on the strided views; a plain silu does not)
                h = ggml_swiglu_split(gc, g, u);
                ggml_tensor * y0 = ggml_mul_mat(gc, ss.down[w], h);
                if (s_down[w] != 1.0f) {
                    y0 = ggml_scale(gc, y0, s_down[w]);
                }
                ggml_build_forward_expand(gf, ggml_set_rows(gc, out2, y0, sv));
                ++visited;
                continue;
            }
            h = ggml_mul_mat(gc, ss.up[w], x);
            if (s_up[w] != 1.0f) {
                h = ggml_scale(gc, h, s_up[w]);
            }
            if (ss.gate[w]) {
                ggml_tensor * g = ggml_mul_mat(gc, ss.gate[w], x);
                if (s_gate[w] != 1.0f) {
                    g = ggml_scale(gc, g, s_gate[w]);
                }
                if (swiglu_limit > 1e-6f) {
                    // build_moe_ffn's generic clamped SwiGLU
                    ggml_tensor * uc = ggml_clamp(gc, h, -swiglu_limit, swiglu_limit);
                    ggml_tensor * ga = ggml_clamp(gc, ggml_silu(gc, g), -INFINITY, swiglu_limit);
                    h = ggml_mul(gc, ga, uc);
                } else {
                    h = ggml_swiglu_split(gc, g, h);
                }
            } else if (act_ == act::relu_sqr) {
                ggml_tensor * r = ggml_relu(gc, h);
                h = ggml_mul(gc, r, r);
            }
            ggml_tensor * y  = ggml_mul_mat(gc, ss.down[w], h);
            if (s_down[w] != 1.0f) {
                y = ggml_scale(gc, y, s_down[w]);
            }
            ggml_build_forward_expand(gf, ggml_set_rows(gc, out2, y, sv));
            ++visited;
        }

        if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
            ggml_free(gc);
            throw std::runtime_error("moe: expert-major could not allocate its wave graph");
        }
        ggml_backend_graph_compute(backend_, gf);   // synchronous: the uploads above are done too
        for (const ExpertDescriptor * d : held) {
            cache->release_bundle(*d);
        }
        ggml_free(gc);
        ++launches_;
    }
    for (auto & t : prefetchers) { t.join(); }

    experts_visited_ += visited;
    return visited;
}

} // namespace moe
