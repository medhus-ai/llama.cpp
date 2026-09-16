#pragma once

// moe-stream-lab: compact expert pool glue between llama.cpp and the generic expert runtime.
//
// This file is the "llama.cpp MoE adapter": it knows that llama_layer holds ffn_{gate,up,down}_exps with the
// expert axis in ne[2], builds the ExpertIndex from a loaded model, replaces those tensors by pool tensors
// (ne[2] == n_slots) and, through the scheduler eval callback, fills slots on demand right before the
// MUL_MAT_ID nodes run. See moe-stream-lab/docs/decisions/0003-m3-compact-pool-mechanism.md.

#include "moe-runtime/expert-runtime.h"

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
    llama_moe_pool(llama_model & model, int32_t n_slots, int32_t store_mode, const std::string & model_path, bool verify, bool shared, int32_t io_threads, const std::string & pack_path, uint64_t host_cache_bytes,
                   int32_t prefetch_k, int32_t prefetch_lookahead, const char * prefetch_src, const char * tier_state);
    ~llama_moe_pool();

    llama_moe_pool(const llama_moe_pool &) = delete;
    llama_moe_pool & operator=(const llama_moe_pool &) = delete;

    // scheduler eval callback; user_data == llama_moe_pool*. Chains to `user_cb` if set.
    static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

    std::string stats_json() const;

    ggml_backend_sched_eval_callback user_cb = nullptr;
    void *                           user_ud = nullptr;

    int32_t n_slots = 0;
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
        std::string src_prefix;     // e.g. "nemotron_h_block_out-"
        struct layer_t { uint32_t il; std::vector<float> norm, gate, bias; uint32_t n_embd = 0, n_expert = 0; };
        std::vector<layer_t> layers;             // by index_.moe_layers order
        std::vector<float>   x;                  // scratch: [n_tok, n_embd]
        std::vector<float>   logits;             // scratch
        uint64_t predictions = 0, issued = 0, dropped = 0;   // dropped: queue was full, prediction discarded
        // background queue: the callback only enqueues, never waits
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<std::pair<uint32_t, uint32_t>> queue;   // (layer, expert)
        bool stop = false;
        std::vector<std::thread> threads;
    } prerouter_;
    void prefetch_loop();
    void init_prerouter(llama_model & model, int k, int lookahead, const std::string & src_prefix);
    void on_residual(struct ggml_tensor * t, uint32_t block);
    void on_prefetch_ids(struct ggml_tensor * t, uint32_t layer);   // in-graph prerouter output
    void on_prefetch_probs(struct ggml_tensor * t, uint32_t layer, bool entry);
    void issue_prefetch(uint32_t il_t, const std::vector<int32_t> & ids, const std::vector<float> * probs, int64_t n_tok);
    float prefetch_margin_ = 0.0f;   // item 2: keep candidate i only if prob_i >= prob_of_(n_used)th * (1 - margin); 0 = off
    std::vector<int32_t> pending_ids_;   // ids seen, waiting for their probs tensor (same callback batch)
    uint32_t pending_layer_ = UINT32_MAX; bool pending_entry_ = false; int64_t pending_ntok_ = 0;
    bool prerouter_in_graph_ = true;

    // tier warm-start: the resident set is written to tier_state_path_ on exit and read back on the next start
    std::string       tier_state_path_;
    std::thread       warm_thread_;
    std::atomic<bool> warm_stop_{false};
    uint64_t          warm_requested_ = 0, warm_loaded_ = 0;
    void warm_start();
    void save_tier_state();

    bool    shared_ = false;

private:
    void on_slot_ids(struct ggml_tensor * t, uint32_t il);

    moe::ExpertIndex                             index_;
    std::unique_ptr<moe::ExpertStore>            store_;
    moe::CachingExpertStore *                    host_cache_ = nullptr;   // non-owning view into store_ when enabled
    std::vector<std::unique_ptr<moe::LayerPool>> pools_;   // by index_.layer_slot(il), or one shared pool
    std::map<std::string, ggml_tensor *>         shared_tensors_;
    moe::PoolStats                               stats_;
    uint64_t                                     clock_ = 0;

    // one ggml context + buffer per buffer type (layers may live on different devices)
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_by_buft_;
    std::vector<ggml_backend_buffer_t> bufs_;

    std::vector<int32_t>        ids_buf_;
    std::vector<moe::ExpertKey> keys_buf_;
    std::vector<int32_t>        slots_buf_;
};
