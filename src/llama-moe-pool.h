#pragma once

// moe-stream-lab: compact expert pool glue between llama.cpp and the generic expert runtime.
//
// This file is the "llama.cpp MoE adapter": it knows that llama_layer holds ffn_{gate,up,down}_exps with the
// expert axis in ne[2], builds the ExpertIndex from a loaded model, replaces those tensors by pool tensors
// (ne[2] == n_slots) and, through the scheduler eval callback, fills slots on demand right before the
// MUL_MAT_ID nodes run. See moe-stream-lab/docs/decisions/0003-m3-compact-pool-mechanism.md.

#include "moe-runtime/expert-runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct ggml_context;

struct llama_moe_pool {
    // store_mode: 0 = resident tensors (memory), 1 = positional reads from the model file
    llama_moe_pool(llama_model & model, int32_t n_slots, int32_t store_mode, const std::string & model_path, bool verify);
    ~llama_moe_pool();

    llama_moe_pool(const llama_moe_pool &) = delete;
    llama_moe_pool & operator=(const llama_moe_pool &) = delete;

    // scheduler eval callback; user_data == llama_moe_pool*. Chains to `user_cb` if set.
    static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

    std::string stats_json() const;

    ggml_backend_sched_eval_callback user_cb = nullptr;
    void *                           user_ud = nullptr;

    int32_t n_slots = 0;

private:
    void on_slot_ids(struct ggml_tensor * t, uint32_t il);

    moe::ExpertIndex                             index_;
    std::unique_ptr<moe::ExpertStore>            store_;
    std::vector<std::unique_ptr<moe::LayerPool>> pools_;   // by index_.layer_slot(il)
    moe::PoolStats                               stats_;
    uint64_t                                     clock_ = 0;

    ggml_context *        ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;

    std::vector<int32_t>        ids_buf_;
    std::vector<moe::ExpertKey> keys_buf_;
    std::vector<int32_t>        slots_buf_;
};
