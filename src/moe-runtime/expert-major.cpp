#include "expert-major.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace moe {

static int find_slice(const ExpertDescriptor & e, const char * want) {
    for (size_t i = 0; i < e.slices.size(); ++i) {
        if (e.slices[i].name.find(want) != std::string::npos) {
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
    if ((slice_up_ < 0 && slice_gate_up_ < 0) || slice_down_ < 0) {
        throw std::runtime_error("moe: expert-major needs a down slice and either up or gate_up");
    }
    build_slots(proto);
}

void ExpertMajorFFN::build_slots(const ExpertDescriptor & proto) {
    // Worst case every (rank, token) pair lands on one expert, so index buffers are sized for the
    // largest chunk we allow. 8192 pairs covers a 1024-token chunk at top-8.
    max_rows_ = 65536;   // 8192 tokens at top-8

    const size_t n_tensors = (size_t) wave_ * 3 + 8;
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors, nullptr, /*no_alloc*/ true };
    ctx_ = ggml_init(ip);

    const auto & sd = proto.slices[slice_down_];

    w_up_.assign((size_t) wave_, nullptr);
    w_down_.resize((size_t) wave_);
    w_gate_.assign((size_t) wave_, nullptr);
    w_gate_up_.assign((size_t) wave_, nullptr);
    for (int64_t w = 0; w < wave_; ++w) {
        if (slice_up_ >= 0) {
            w_up_[(size_t) w] = ggml_new_tensor_2d(ctx_, proto.slices[slice_up_].type, n_embd_, n_ff_);
        }
        w_down_[(size_t) w] = ggml_new_tensor_2d(ctx_, sd.type, n_ff_,  n_embd_);
        if (slice_gate_up_ >= 0) {
            w_gate_up_[(size_t) w] = ggml_new_tensor_2d(ctx_, proto.slices[slice_gate_up_].type, n_embd_, 2 * n_ff_);
        }
        if (slice_gate_ >= 0) {
            w_gate_[(size_t) w] = ggml_new_tensor_2d(ctx_, proto.slices[slice_gate_].type, n_embd_, n_ff_);
        }
    }
    gidx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, max_rows_);
    sidx_ = ggml_new_tensor_1d(ctx_, GGML_TYPE_I64, max_rows_);

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buf_) {
        throw std::runtime_error("moe: expert-major could not allocate its expert slot");
    }
    galloc_    = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    // one chain of ~8 nodes per expert in a wave, plus slack
    graph_mem_.resize(ggml_tensor_overhead() * (size_t) (wave_ * 12 + 32) + ggml_graph_overhead_custom(wave_ * 12 + 32, false));

    size_t widest = sd.bytes;
    if (slice_up_ >= 0) {
        widest = std::max(widest, proto.slices[slice_up_].bytes);
    }
    if (slice_gate_up_ >= 0) {
        widest = std::max(widest, proto.slices[slice_gate_up_].bytes);
    }
    if (slice_gate_ >= 0) {
        widest = std::max(widest, proto.slices[slice_gate_].bytes);
    }
    staging_.resize(widest);
    by_expert_.resize((size_t) n_expert_);
}

ExpertMajorFFN::~ExpertMajorFFN() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (buf_)    ggml_backend_buffer_free(buf_);
    if (ctx_)    ggml_free(ctx_);
}

size_t ExpertMajorFFN::run(ExpertStore & store, const ExpertIndex & index, uint32_t layer,
                           const int32_t * ids, int64_t n_tokens, ggml_tensor * inp, ggml_tensor * out) {
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
            if (rows32_[r] < 0 || rows32_[r] >= (int32_t) inp->ne[1]) {
                fprintf(stderr, "moe: expert-major gather row %d out of [0,%lld) (layer %u)\n",
                        rows32_[r], (long long) inp->ne[1], layer);
                throw std::runtime_error("moe: expert-major gather index out of range");
            }
            if (rows64_[r] < 0 || rows64_[r] >= out->ne[1]) {
                fprintf(stderr, "moe: expert-major scatter row %lld out of [0,%lld) (layer %u)\n",
                        (long long) rows64_[r], (long long) out->ne[1], layer);
                throw std::runtime_error("moe: expert-major scatter index out of range");
            }
        }
        ggml_backend_tensor_set(gidx_, rows32_.data(), 0, rows32_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(sidx_, rows64_.data(), 0, rows64_.size() * sizeof(int64_t));

        for (size_t w = 0; w < n_in_wave; ++w) {
            const ExpertDescriptor & d = index.get(ExpertKey{ layer, (uint32_t) used[base + w] });
            auto load = [&](int slice, ggml_tensor * dst) {
                if (slice < 0 || !dst) {
                    return;
                }
                const auto & sl = d.slices[slice];
                if (sl.bytes > ggml_nbytes(dst) || sl.bytes > staging_.size()) {
                    throw std::runtime_error("moe: expert-major slot does not match this layer's expert slice; "
                                             "a dynamic quant probably changes the expert type per layer");
                }
                store.read_slice(d, (size_t) slice, staging_.data());
                ggml_backend_tensor_set(dst, staging_.data(), 0, sl.bytes);
                bytes_fetched_ += sl.bytes;
            };
            load(slice_gate_up_, w_gate_up_[w]);
            load(slice_gate_, w_gate_[w]);
            load(slice_up_,   w_up_[w]);
            load(slice_down_, w_down_[w]);
        }

        ggml_init_params gp = { graph_mem_.size(), graph_mem_.data(), /*no_alloc*/ true };
        ggml_context * gc = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph_custom(gc, (size_t) (wave_ * 12 + 32), false);

        for (size_t w = 0; w < n_in_wave; ++w) {
            const int64_t m = counts[w];
            ggml_tensor * gv = ggml_view_1d(gc, gidx_, m, (size_t) starts[w] * sizeof(int32_t));
            ggml_tensor * sv = ggml_view_1d(gc, sidx_, m, (size_t) starts[w] * sizeof(int64_t));
            ggml_tensor * x  = ggml_get_rows(gc, inp, gv);
            ggml_tensor * h  = nullptr;
            if (w_gate_up_[w]) {
                // one GEMM, then split: gate = rows [0, n_ff), up = rows [n_ff, 2*n_ff)
                ggml_tensor * gu = ggml_mul_mat(gc, w_gate_up_[w], x);   // [2*n_ff, m]
                ggml_tensor * g  = ggml_view_2d(gc, gu, n_ff_, gu->ne[1], gu->nb[1], 0);
                ggml_tensor * u  = ggml_view_2d(gc, gu, n_ff_, gu->ne[1], gu->nb[1], n_ff_ * gu->nb[0]);
                h = ggml_mul(gc, ggml_silu(gc, g), u);
                ggml_tensor * y0 = ggml_mul_mat(gc, w_down_[w], h);
                ggml_build_forward_expand(gf, ggml_set_rows(gc, out, y0, sv));
                ++visited;
                continue;
            }
            h = ggml_mul_mat(gc, w_up_[w], x);
            if (w_gate_[w]) {
                ggml_tensor * g = ggml_mul_mat(gc, w_gate_[w], x);
                h = ggml_mul(gc, ggml_silu(gc, g), h);
            } else if (act_ == act::relu_sqr) {
                ggml_tensor * r = ggml_relu(gc, h);
                h = ggml_mul(gc, r, r);
            }
            ggml_tensor * y  = ggml_mul_mat(gc, w_down_[w], h);
            ggml_build_forward_expand(gf, ggml_set_rows(gc, out, y, sv));
            ++visited;
        }

        if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
            ggml_free(gc);
            throw std::runtime_error("moe: expert-major could not allocate its wave graph");
        }
        ggml_backend_graph_compute(backend_, gf);
        ggml_free(gc);
        ++launches_;
    }

    experts_visited_ += visited;
    return visited;
}

} // namespace moe
