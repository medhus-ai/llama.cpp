// ADR 0013: expert-major MoE FFN must agree with the token-major MUL_MAT_ID path.
//
// build_moe_ffn reduces the per-rank outputs in rank order, so the order experts are visited in
// cannot change the result -- provided the two paths compute the same products. mul_mat and
// mul_mat_id are separate kernels, so that has to be measured, not assumed.

#include "moe-runtime/expert-major.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Serves expert bytes straight out of a host buffer, so the test does not depend on any store.
class BufferStore : public moe::ExpertStore {
public:
    BufferStore(const std::vector<uint8_t> & blob) : blob_(blob) {}
    void read_slice(const moe::ExpertDescriptor & e, size_t slice, void * dst) override {
        const auto & sl = e.slices[slice];
        memcpy(dst, blob_.data() + sl.offset, sl.bytes);
    }
    const char * name() const override { return "test-buffer"; }
private:
    const std::vector<uint8_t> & blob_;
};

int main() {
    const int64_t n_embd = 512, n_ff = 256, n_expert = 16, n_used = 4, n_tokens = 64;
    const ggml_type wt = GGML_TYPE_Q4_0;   // 32-wide block, divides both dims

    ggml_backend_t be = ggml_backend_cpu_init();
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);

    const size_t row_e = ggml_row_size(wt, n_embd);   // rows of length n_embd (gate, up)
    const size_t row_f = ggml_row_size(wt, n_ff);     // rows of length n_ff   (down)
    const size_t gate_b = row_e * n_ff, up_b = row_e * n_ff, down_b = row_f * n_embd;
    const size_t bundle = gate_b + up_b + down_b;

    // one blob holding every expert's gate|up|down, which is also the layout the descriptors index
    std::vector<uint8_t> blob(bundle * n_expert);
    {
        std::vector<float> t1((size_t) n_embd * n_ff), t2((size_t) n_ff * n_embd);
        for (int64_t e = 0; e < n_expert; ++e) {
            uint8_t * base = blob.data() + (size_t) e * bundle;
            for (auto & v : t1) v = uni(rng);
            ggml_quantize_chunk(wt, t1.data(), base,                    0, n_ff,   n_embd, nullptr);
            for (auto & v : t1) v = uni(rng);
            ggml_quantize_chunk(wt, t1.data(), base + gate_b,           0, n_ff,   n_embd, nullptr);
            for (auto & v : t2) v = uni(rng);
            ggml_quantize_chunk(wt, t2.data(), base + gate_b + up_b,    0, n_embd, n_ff,   nullptr);
        }
    }

    moe::ExpertIndex index;
    index.n_expert = (uint32_t) n_expert;
    index.moe_layers = { 0 };
    index.descs.resize((size_t) n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        moe::ExpertDescriptor d;
        d.key = { 0, (uint32_t) e };
        const uint64_t base = (uint64_t) e * bundle;
        d.slices.push_back({ "blk.0.ffn_gate_exps.weight", nullptr, 0, base,                   gate_b, wt });
        d.slices.push_back({ "blk.0.ffn_up_exps.weight",   nullptr, 0, base + gate_b,          up_b,   wt });
        d.slices.push_back({ "blk.0.ffn_down_exps.weight", nullptr, 0, base + gate_b + up_b,   down_b, wt });
        d.total_bytes = bundle;
        index.descs[(size_t) e] = std::move(d);
    }

    const int64_t n_pairs = n_tokens * n_used;
    std::vector<float>   xf((size_t) n_embd * n_tokens);
    for (auto & v : xf) v = uni(rng);
    std::vector<int32_t> ids((size_t) n_pairs);
    for (auto & v : ids) v = (int32_t) (rng() % n_expert);

    ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * inp  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * outA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_pairs);
    ggml_tensor * outB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_pairs);
    ggml_tensor * as_g = ggml_new_tensor_3d(ctx, wt, n_embd, n_ff,  n_expert);
    ggml_tensor * as_u = ggml_new_tensor_3d(ctx, wt, n_embd, n_ff,  n_expert);
    ggml_tensor * as_d = ggml_new_tensor_3d(ctx, wt, n_ff,  n_embd, n_expert);
    ggml_tensor * idt  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    GGML_ASSERT(buf);

    ggml_backend_tensor_set(inp, xf.data(),  0, ggml_nbytes(inp));
    ggml_backend_tensor_set(idt, ids.data(), 0, ggml_nbytes(idt));
    for (int64_t e = 0; e < n_expert; ++e) {
        const uint8_t * base = blob.data() + (size_t) e * bundle;
        ggml_backend_tensor_set(as_g, base,                  (size_t) e * gate_b, gate_b);
        ggml_backend_tensor_set(as_u, base + gate_b,         (size_t) e * up_b,   up_b);
        ggml_backend_tensor_set(as_d, base + gate_b + up_b,  (size_t) e * down_b, down_b);
    }

    // ---- token-major reference: one MUL_MAT_ID per projection over the whole expert set --------
    {
        std::vector<uint8_t> gmem(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
        ggml_init_params gp = { gmem.size(), gmem.data(), true };
        ggml_context * gc = ggml_init(gp);
        ggml_tensor * x3 = ggml_reshape_3d(gc, inp, n_embd, 1, n_tokens);
        ggml_tensor * g  = ggml_mul_mat_id(gc, as_g, x3, idt);
        ggml_tensor * u  = ggml_mul_mat_id(gc, as_u, x3, idt);
        ggml_tensor * h  = ggml_mul(gc, ggml_silu(gc, g), u);
        ggml_tensor * y  = ggml_mul_mat_id(gc, as_d, h, idt);
        ggml_tensor * cp = ggml_cpy(gc, ggml_reshape_2d(gc, y, n_embd, n_pairs), outA);
        ggml_cgraph * gf = ggml_new_graph(gc);
        ggml_build_forward_expand(gf, cp);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        ggml_backend_graph_compute(be, gf);
        ggml_gallocr_free(ga);
        ggml_free(gc);
    }

    // ---- expert-major ---------------------------------------------------------------------
    BufferStore store(blob);
    moe::ExpertMajorFFN em(be, index.descs[0], n_embd, n_ff, n_expert, n_used,
                           moe::ExpertMajorFFN::act::silu);
    const size_t visited = em.run(store, index, 0, ids.data(), n_tokens, inp, outB);

    std::vector<float> a((size_t) n_embd * n_pairs), b((size_t) n_embd * n_pairs);
    ggml_backend_tensor_get(outA, a.data(), 0, ggml_nbytes(outA));
    ggml_backend_tensor_get(outB, b.data(), 0, ggml_nbytes(outB));

    size_t diff = 0;
    double maxabs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++diff;
            const double d = fabs((double) a[i] - (double) b[i]);
            if (d > maxabs) maxabs = d;
        }
    }
    printf("experts visited : %zu / %lld\n", visited, (long long) n_expert);
    printf("bytes fetched   : %llu (bundle %zu x %zu experts)\n",
           (unsigned long long) em.bytes_fetched(), bundle, visited);
    printf("values compared : %zu\n", a.size());
    printf("bitwise differs : %zu\n", diff);
    printf("max |delta|     : %.3e\n", maxabs);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);

    if (diff != 0) {
        printf("FAIL: expert-major does not match MUL_MAT_ID\n");
        return 1;
    }
    printf("ok: expert-major is bit-exact against MUL_MAT_ID\n");
    return 0;
}
