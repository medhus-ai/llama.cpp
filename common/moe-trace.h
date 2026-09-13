#pragma once

// MoE routing tracer (moe-stream-lab, milestone M1).
//
// Non-invasive observation of routed-expert selection: installs a ggml_backend_sched eval callback that
// reads the top-k expert id tensors ("ffn_moe_topk-<il>") produced by build_moe_ffn after they are
// computed, and appends one JSON line per (token, layer):
//
//     {"token": 121, "layer": 17, "experts": [3, 19, 40, 66, 89, 118]}
//
// The first line is {"meta": {...}} with n_expert / n_expert_used as observed from the graph (nothing is
// hard-coded). Token indices are sequential per process (prompt tokens first, then generated tokens).
//
// When no trace file is configured, nothing is installed and execution is unchanged.

#include "ggml-backend.h"

#include <string>

// Open the trace file and return the callback user-data pointer (process-wide singleton).
// Returns nullptr on failure. Tracing starts disabled; call common_moe_trace_set_enabled(true) after warm-up.
void * common_moe_trace_install(const std::string & path);

bool common_moe_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

void common_moe_trace_set_enabled(bool enabled);

// Operator-level verification (moe-stream-lab spec §10): dump every "ffn_moe_out-<il>" tensor of the first
// `n_ubatches` ubatches as raw F32 files "<dir>/ub<k>_layer<il>.f32" (header: two uint32 ne[0], ne[1]).
// Same prompt in two modes -> compare with moe-stream-lab/scripts/compare_moe_out.py.
void common_moe_trace_set_dump(const std::string & dir, int n_ubatches);
// Optional: dump tensors whose name (before the -<il> suffix) matches this prefix instead of ffn_moe_out.
void common_moe_trace_set_dump_prefix(const std::string & prefix);

// Flush and close (also called automatically at exit).
void common_moe_trace_close();
