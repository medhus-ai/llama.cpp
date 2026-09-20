#pragma once

// moe-stream-lab: expert-major MoE FFN for prefill (ADR 0013).
//
// Token-major prefill runs one MUL_MAT_ID over the pool, so every expert a ubatch selects must be
// resident at the same time. That is what forces n_ubatch <= n_slots / n_expert_used and keeps
// prefill at a few rows per expert GEMM.
//
// Expert-major inverts the loop: for one chunk of tokens, visit each expert once, gather the rows
// routed to it, run gate/up/act/down on them, and scatter the result back. Only one expert needs to
// be resident (two when double buffered), so the chunk is no longer bounded by the pool.
//
// Reduction order is unchanged: results are written per (rank, token) slot and summed later in rank
// order by build_moe_ffn, so the order experts are visited in cannot matter. Measured bit-exact on
// CPU against MUL_MAT_ID over 7 shapes; ~4e-5 relative on CUDA, where mul_mat and mul_mat_id pick
// different kernels.

#include "expert-runtime.h"

#include <array>
#include <cstdint>
#include <map>
#include <vector>

struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;

namespace moe {

class ExpertMajorFFN {
public:
    // gating_op mirrors llama_expert_gating_func_type; only SiLU and ReLU^2 are needed so far.
    enum class act { silu, relu_sqr };

    // n_ff is the per-expert intermediate size. proto supplies the slice types and byte sizes; every
    // expert in a layer shares them.
    // wave = how many experts are resident at once. Each wave is one graph launch, so this trades
    // a little memory (one bundle per expert, ~1 MiB) for far fewer submissions.
    ExpertMajorFFN(ggml_backend_t backend, const ExpertDescriptor & proto,
                   int64_t n_embd, int64_t n_ff, int64_t n_expert, int64_t n_expert_used, act a,
                   int64_t wave = 32);
    ~ExpertMajorFFN();

    ExpertMajorFFN(const ExpertMajorFFN &) = delete;
    ExpertMajorFFN & operator=(const ExpertMajorFFN &) = delete;

    // ids: host side, n_expert_used * n_tokens, values in [0, n_expert).
    // inp: device [n_embd, n_tokens]. The FFN input is the same for every rank of a token, so the
    //      gather reads row p / n_expert_used while the scatter writes row p.
    // out: device [n_embd, n_expert_used * n_tokens], one row per (rank, token); caller reduces.
    // Returns the number of experts actually visited.
    // swiglu_limit > 0 reproduces build_moe_ffn's clamped SwiGLU (hparams.swiglu_clamp_exp[il]).
    size_t run(ExpertStore & store, const ExpertIndex & index, uint32_t layer,
               const int32_t * ids, int64_t n_tokens, ggml_tensor * inp, ggml_tensor * out,
               float swiglu_limit = 0.0f);

    uint64_t experts_visited() const { return experts_visited_; }
    uint64_t bytes_fetched()   const { return bytes_fetched_; }
    uint64_t launches()        const { return launches_; }

private:
    ggml_backend_t        backend_ = nullptr;
    ggml_context *        ctx_     = nullptr;   // index buffers
    ggml_backend_buffer_t buf_     = nullptr;
    ggml_gallocr_t        galloc_  = nullptr;

    // `wave_` resident experts: gate, up, down each. gate may be absent (ReLU^2 architectures).
    // Dynamic quants (Unsloth UD) change the expert tensor types from layer to layer, so the slots
    // are kept per type set and picked by the layer being run; a few sets of wave x 3 tensors.
    struct slot_set {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        std::vector<ggml_tensor *> gate_up;   // fused gate+up (Gemma 4): [n_embd, 2*n_ff]
        std::vector<ggml_tensor *> gate;
        std::vector<ggml_tensor *> up;
        std::vector<ggml_tensor *> down;
    };
    using type_key = std::array<int, 4>;   // gate, up, gate_up, down types (GGML_TYPE_COUNT = absent)
    std::map<type_key, slot_set> sets_;
    slot_set & slots_for(const ExpertDescriptor & d);
    void build_index_buffers();
    int64_t wave_ = 32;
    uint64_t launches_ = 0;
    ggml_tensor * gidx_   = nullptr;   // I32 gather rows
    ggml_tensor * sidx_   = nullptr;   // I64 scatter rows

    int slice_gate_ = -1, slice_up_ = -1, slice_down_ = -1, slice_gate_up_ = -1;
    // per-expert f32 scales (dynamic quants such as Gemma 4 ship ffn_down_exps.scale); -1 when absent.
    // Applied exactly as build_lora_mm_id does: the whole gate_up product by the up scale, the
    // down product by the down scale.
    int slice_gate_s_ = -1, slice_up_s_ = -1, slice_down_s_ = -1;
    float read_scale(ExpertStore & store, const ExpertDescriptor & d, int slice);

    int64_t n_embd_ = 0, n_ff_ = 0, n_expert_ = 0, n_used_ = 0, max_rows_ = 0;
    act     act_    = act::silu;

    std::vector<uint8_t>  staging_;
    std::vector<int32_t>  rows32_;
    std::vector<int64_t>  rows64_;
    std::vector<std::vector<int32_t>> by_expert_;
    std::vector<uint8_t>  graph_mem_;

    uint64_t experts_visited_ = 0;
    uint64_t bytes_fetched_   = 0;
};

} // namespace moe
