#include "moe-trace.h"

#include "ggml.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct moe_tracer {
    FILE *  f            = nullptr;
    bool    enabled      = false;
    bool    meta_written = false;
    int64_t n_expert     = -1;   // learned from ffn_moe_probs-<il> (ne[0])
    int64_t n_expert_used= -1;   // learned from ffn_moe_topk-<il>  (ne[0])
    int64_t token_base   = 0;    // index of the first token of the current ubatch
    int64_t last_il      = -1;   // layer of the previous topk observation
    int64_t last_n_tok   = 0;    // n_tokens of the previous ubatch
    std::vector<int32_t> buf;
    std::mutex mtx;
};

moe_tracer & tracer() {
    static moe_tracer t;
    return t;
}

// parse "<prefix>-<il>" ; returns il or -1
int parse_layer(const char * name, const char * prefix, size_t prefix_len) {
    if (strncmp(name, prefix, prefix_len) != 0) {
        return -1;
    }
    const char * p = name + prefix_len;
    if (*p == '\0') {
        return -1;
    }
    for (const char * q = p; *q; ++q) {
        if (*q < '0' || *q > '9') {
            return -1;
        }
    }
    return atoi(p);
}

constexpr const char * TOPK_PREFIX  = "ffn_moe_topk-";
constexpr const char * PROBS_PREFIX = "ffn_moe_probs-";

} // namespace

void * common_moe_trace_install(const std::string & path) {
    auto & t = tracer();
    std::lock_guard<std::mutex> lock(t.mtx);
    if (t.f) {
        return &t;
    }
    t.f = fopen(path.c_str(), "w");
    if (!t.f) {
        LOG_ERR("%s: failed to open MoE trace file '%s'\n", __func__, path.c_str());
        return nullptr;
    }
    atexit(common_moe_trace_close);
    LOG_INF("%s: MoE routing trace -> %s\n", __func__, path.c_str());
    return &t;
}

void common_moe_trace_set_enabled(bool enabled) {
    auto & t = tracer();
    std::lock_guard<std::mutex> lock(t.mtx);
    t.enabled = enabled;
    // a fresh start: tokens are counted from zero after warm-up
    t.token_base = 0;
    t.last_il    = -1;
    t.last_n_tok = 0;
}

void common_moe_trace_close() {
    auto & t = tracer();
    std::lock_guard<std::mutex> lock(t.mtx);
    if (t.f) {
        fclose(t.f);
        t.f = nullptr;
    }
}

bool common_moe_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * tr = (moe_tracer *) user_data;

    const int il_topk  = parse_layer(t->name, TOPK_PREFIX,  strlen(TOPK_PREFIX));
    const int il_probs = il_topk >= 0 ? -1 : parse_layer(t->name, PROBS_PREFIX, strlen(PROBS_PREFIX));

    if (ask) {
        // only request data for the tensors we care about; everything else runs fused/async as usual
        return il_topk >= 0 || (il_probs >= 0 && tr->n_expert < 0);
    }

    if (!tr->f) {
        return true;
    }

    std::lock_guard<std::mutex> lock(tr->mtx);

    if (il_probs >= 0) {
        if (tr->n_expert < 0) {
            tr->n_expert = t->ne[0];
        }
        return true;
    }

    if (il_topk < 0) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_I32 && "ffn_moe_topk is expected to be I32");

    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];

    if (tr->n_expert_used < 0) {
        tr->n_expert_used = n_used;
    }

    if (!tr->enabled) {
        return true; // warm-up or explicitly paused
    }

    // new ubatch: the layer index went back (or stayed) relative to the previous observation
    if (il_topk <= tr->last_il) {
        tr->token_base += tr->last_n_tok;
    }
    tr->last_il    = il_topk;
    tr->last_n_tok = n_tok;

    if (!tr->meta_written) {
        fprintf(tr->f, "{\"meta\": {\"n_expert\": %lld, \"n_experts_used\": %lld}}\n",
                (long long) tr->n_expert, (long long) n_used);
        tr->meta_written = true;
    }

    const size_t n_bytes = ggml_nbytes(t);
    tr->buf.resize(n_bytes / sizeof(int32_t));
    ggml_backend_tensor_get(t, tr->buf.data(), 0, n_bytes);

    const size_t nb1 = t->nb[1] / sizeof(int32_t);
    for (int64_t i = 0; i < n_tok; ++i) {
        fprintf(tr->f, "{\"token\": %lld, \"layer\": %d, \"experts\": [",
                (long long) (tr->token_base + i), il_topk);
        for (int64_t k = 0; k < n_used; ++k) {
            fprintf(tr->f, "%s%d", k ? ", " : "", tr->buf[i * nb1 + k]);
        }
        fputs("]}\n", tr->f);
    }
    return true;
}
