#pragma once

// moe-stream-lab: compact expert pool glue between llama.cpp and the generic expert runtime.
//
// This file is the "llama.cpp MoE adapter": it knows that llama_layer holds ffn_{gate,up,down}_exps with the
// expert axis in ne[2], builds the ExpertIndex from a loaded model, replaces those tensors by pool tensors
// (ne[2] == n_slots) and, through the scheduler eval callback, fills slots on demand right before the
// MUL_MAT_ID nodes run. See moe-stream-lab/docs/decisions/0003-m3-compact-pool-mechanism.md.

#include "moe-runtime/expert-runtime.h"
#include "moe-runtime/expert-major.h"

#include <cstdint>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <string>
#include <vector>

struct llama_model;
struct ggml_context;

struct llama_moe_pool {
    // store_mode: 0 = resident tensors (memory), 1 = positional reads from the model file
    llama_moe_pool(llama_model & model, int32_t n_slots, int32_t store_mode, const std::string & model_path,
                   bool verify, int32_t io_threads, uint64_t host_cache_bytes, int32_t prefetch_k);
    ~llama_moe_pool();

    llama_moe_pool(const llama_moe_pool &) = delete;
    llama_moe_pool & operator=(const llama_moe_pool &) = delete;

    // scheduler eval callback; user_data == llama_moe_pool*. Chains to `user_cb` if set.
    static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

    std::string stats_json() const;

    // --- expert-major prefill (ADR 0013) -------------------------------------------------------
    // When on and a ubatch is at least em_min_tokens, build_moe_ffn skips the MUL_MAT_ID chain and
    // emits a view of em_out(il) instead. The eval callback then computes that layer's FFN
    // expert-major, visiting each expert once, so the chunk size stops depending on the pool.
    bool          em_enabled   = false;
    int32_t       em_min_tokens = 32;
    // Use expert-major at or above the threshold, and also below it whenever the pool could not
    // serve that ubatch token-major anyway (a remainder ubatch after a long prefill, for example).
    // Without the second clause a 3-token remainder would fall back and abort on a small pool.
    bool          em_active(int64_t n_tokens) const {
        if (!em_enabled) { return false; }
        return n_tokens >= em_min_tokens || n_tokens * em_n_used > (int64_t) n_slots;
    }
    int64_t       em_n_used = 0;
    ggml_tensor * em_out(uint32_t il);                       // persistent [n_embd, n_used, n_ubatch]
    void          em_register(uint32_t il, ggml_tensor * inp);  // FFN input for this layer, this graph
    void          em_init(llama_model & model, uint32_t n_ubatch, int32_t wave);
    uint64_t      em_layers_run = 0, em_experts_visited = 0, em_bytes = 0;

    ggml_backend_sched_eval_callback user_cb = nullptr;
    void *                           user_ud = nullptr;

    int32_t n_slots = 0;
    struct em_layer_t {
        ggml_tensor * out = nullptr;   // pool-owned output buffer (shared by every layer)
        ggml_tensor * inp = nullptr;   // graph-owned FFN input, valid during execution
    };
    std::map<uint32_t, em_layer_t> em_;
    // One engine for every layer: run() takes the layer, and only one layer computes at a time, so
    // per-layer engines would duplicate the wave buffers (48 x 57 MiB on Qwen3.8).
    std::unique_ptr<moe::ExpertMajorFFN> em_ffn_;
    ggml_context *        em_ctx_ = nullptr;
    ggml_backend_buffer_t em_buf_ = nullptr;
    // called by llama_context once it knows its backends; enables async H2D for device pools
    void attach_compute_backend(ggml_backend_t compute);
    ggml_backend_t transfer_backend_ = nullptr;   // owned; a second backend instance = its own stream
    ggml_backend_dev_t device_ = nullptr;
    bool async_enabled_ = true;

    // M9: the model's own routers as prerouters. At the residual-stream tensor of block j, run the router of
    // the MoE layer that is `lookahead` MoE layers ahead and pull its top-k non-resident candidates into the
    // host tier in the background. Never touches the VRAM pool, never changes routing.
    struct prerouter_t {
        bool     enabled   = false;
        int      k         = 12;
        int      lookahead = 1;     // in MoE layers
        uint64_t predictions = 0, issued = 0, dropped = 0;   // dropped: queue was full, prediction discarded
        // background queue: the callback only enqueues, never waits
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<std::pair<uint32_t, uint32_t>> queue;   // (layer, expert)
        bool stop = false;
        std::vector<std::thread> threads;
    } prerouter_;
    void prefetch_loop();
    void init_prerouter(llama_model & model, int k);
    void on_prefetch_ids(struct ggml_tensor * t, uint32_t layer, bool entry);   // in-graph prerouter output
    void issue_prefetch(uint32_t il_t, const std::vector<int32_t> & ids, int64_t n_tok);



private:
    void on_slot_ids(struct ggml_tensor * t, uint32_t il);

    moe::ExpertIndex                             index_;
    std::unique_ptr<moe::ExpertStore>            store_;
    moe::CachingExpertStore *                    host_cache_ = nullptr;   // non-owning view into store_ when enabled
    std::vector<std::unique_ptr<moe::LayerPool>> pools_;   // by index_.layer_slot(il), or one shared pool
    moe::PoolStats                               stats_;
    uint64_t                                     clock_ = 0;

    // one ggml context + buffer per buffer type (layers may live on different devices)
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_by_buft_;
    std::vector<ggml_backend_buffer_t> bufs_;

    std::vector<int32_t>        ids_buf_;
    std::vector<moe::ExpertKey> keys_buf_;
    std::vector<int32_t>        slots_buf_;
};
