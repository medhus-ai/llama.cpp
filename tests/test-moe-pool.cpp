// moe-stream-lab: unit tests for the generic expert runtime (src/moe-runtime).
//
// Covers slot-pool semantics that the end-to-end regression cannot isolate:
// byte-exact copies, LRU victim choice, no eviction of in-use slots, stale data can never be served,
// hit/miss accounting, and the hard abort when a ubatch needs more experts than the pool has slots.

#include "moe-runtime/expert-runtime.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static const int64_t NE0 = 64;    // elements per row (F32 keeps the arithmetic trivial)
static const int64_t NE1 = 4;     // rows per expert
static const uint32_t N_EXPERT = 8;
static const uint32_t N_SLOTS  = 3;
static const uint32_t LAYER    = 5;

// Deterministic byte pattern for (expert, kind, index).
static float pattern(uint32_t expert, int kind, int64_t i) {
    return (float) (expert * 1000 + kind * 100) + (float) i * 0.5f;
}

struct fixture {
    ggml_context *        ctx  = nullptr;
    ggml_backend_t        be   = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         src[2] = { nullptr, nullptr };   // two "kinds" (up, down)
    ggml_tensor *         pool[2] = { nullptr, nullptr };
    moe::ExpertIndex      index;

    fixture() {
        be = ggml_backend_cpu_init();
        ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
        ctx = ggml_init(ip);
        for (int k = 0; k < 2; ++k) {
            src[k]  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, N_EXPERT);
            pool[k] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NE1, N_SLOTS);
            ggml_format_name(src[k],  "src%d",  k);
            ggml_format_name(pool[k], "pool%d", k);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        CHECK(buf != nullptr);

        // fill the sources with the known pattern
        std::vector<float> tmp(NE0 * NE1);
        for (int k = 0; k < 2; ++k) {
            for (uint32_t e = 0; e < N_EXPERT; ++e) {
                for (int64_t i = 0; i < NE0 * NE1; ++i) {
                    tmp[i] = pattern(e, k, i);
                }
                ggml_backend_tensor_set(src[k], tmp.data(), (size_t) e * src[k]->nb[2], tmp.size() * sizeof(float));
            }
        }

        index.n_expert = N_EXPERT;
        index.moe_layers = { LAYER };
        index.descs.resize(N_EXPERT);
        for (uint32_t e = 0; e < N_EXPERT; ++e) {
            moe::ExpertDescriptor d;
            d.key = { LAYER, e };
            for (int k = 0; k < 2; ++k) {
                moe::TensorSlice sl;
                sl.name   = ggml_get_name(src[k]);
                sl.source = src[k];
                sl.bytes  = src[k]->nb[2];
                sl.offset = (uint64_t) e * src[k]->nb[2];
                sl.type   = GGML_TYPE_F32;
                d.total_bytes += sl.bytes;
                d.slices.push_back(sl);
            }
            index.descs[e] = d;
        }
    }

    ~fixture() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }

    // read back what the pool holds in `slot` for kind `k`
    std::vector<float> slot_data(int k, int slot) const {
        std::vector<float> out(NE0 * NE1);
        ggml_backend_tensor_get(pool[k], out.data(), (size_t) slot * pool[k]->nb[2], out.size() * sizeof(float));
        return out;
    }

    void check_slot_holds(int slot, uint32_t expert) const {
        for (int k = 0; k < 2; ++k) {
            const auto got = slot_data(k, slot);
            for (int64_t i = 0; i < NE0 * NE1; ++i) {
                CHECK(got[i] == pattern(expert, k, i));
            }
        }
    }

    std::unique_ptr<moe::LayerPool> make_pool() {
        return std::make_unique<moe::LayerPool>(LAYER, N_SLOTS, std::vector<ggml_tensor *>{ pool[0], pool[1] });
    }
};

static std::vector<moe::ExpertKey> keys(std::initializer_list<uint32_t> ids) {
    std::vector<moe::ExpertKey> out;
    for (uint32_t e : ids) {
        out.push_back({ LAYER, e });
    }
    return out;
}

int main() {
    // 1. byte-exact copy and slot ids
    {
        fixture f;
        auto pool = f.make_pool();
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;

        pool->ensure(f.index, store, keys({ 3, 7 }), ids, st, clock);
        CHECK(ids.size() == 2);
        CHECK(ids[0] != ids[1]);
        f.check_slot_holds(ids[0], 3);
        f.check_slot_holds(ids[1], 7);
        CHECK(st.requests == 2 && st.misses == 2 && st.hits == 0);
        CHECK(st.bytes_read == 2 * (uint64_t) f.index.get({ LAYER, 3 }).total_bytes);
        printf("ok: byte-exact copy and distinct slot ids\n");

        // 2. repeats inside one call are deduplicated and map to the same slot
        pool->ensure(f.index, store, keys({ 3, 3, 7 }), ids, st, clock);
        CHECK(ids[0] == ids[1]);
        CHECK(st.hits == 2 && st.misses == 2);  // both were resident
        printf("ok: dedupe within a call, hits counted\n");
    }

    // 3. LRU victim choice
    {
        fixture f;
        auto pool = f.make_pool();
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;

        pool->ensure(f.index, store, keys({ 0 }), ids, st, clock);  // slots: 0
        pool->ensure(f.index, store, keys({ 1 }), ids, st, clock);  // slots: 0,1
        pool->ensure(f.index, store, keys({ 2 }), ids, st, clock);  // slots: 0,1,2 (full)
        pool->ensure(f.index, store, keys({ 0 }), ids, st, clock);  // touch 0 -> 1 is now LRU
        pool->ensure(f.index, store, keys({ 4 }), ids, st, clock);  // must evict expert 1
        const int slot_of_4 = ids[0];
        f.check_slot_holds(slot_of_4, 4);
        CHECK(st.evictions == 1);

        // expert 1 must be gone, expert 0 and 2 must still be resident (no reload)
        const uint64_t before = st.misses;
        pool->ensure(f.index, store, keys({ 0, 2 }), ids, st, clock);
        CHECK(st.misses == before);
        pool->ensure(f.index, store, keys({ 1 }), ids, st, clock);
        CHECK(st.misses == before + 1);
        printf("ok: LRU victim choice, touched entries survive\n");
    }

    // 4. a full working set is never evicted mid-call (no slot is reused for two keys of one ubatch)
    {
        fixture f;
        auto pool = f.make_pool();
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;

        pool->ensure(f.index, store, keys({ 1, 2, 3 }), ids, st, clock);
        CHECK(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
        f.check_slot_holds(ids[0], 1);
        f.check_slot_holds(ids[1], 2);
        f.check_slot_holds(ids[2], 3);
        printf("ok: a full ubatch keeps every expert in its own slot\n");

        // 5. reuse of a slot always refreshes its contents (stale data can never be served)
        pool->ensure(f.index, store, keys({ 4, 5, 6 }), ids, st, clock);
        f.check_slot_holds(ids[0], 4);
        f.check_slot_holds(ids[1], 5);
        f.check_slot_holds(ids[2], 6);
        for (const auto & s : pool->slots()) {
            CHECK(s.state == moe::SlotState::READY);
            CHECK(s.generation >= 1);
        }
        printf("ok: refilled slots hold the new expert, states released\n");
    }

    // 6. hard abort when the working set exceeds the pool (no silent fallback)
    {
        fixture f;
        auto pool = f.make_pool();
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;
        bool threw = false;
        try {
            pool->ensure(f.index, store, keys({ 0, 1, 2, 3 }), ids, st, clock);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        printf("ok: aborts when the ubatch needs more experts than there are slots\n");
    }

    // 7. index bounds and wrong-layer keys are rejected
    {
        fixture f;
        auto pool = f.make_pool();
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;
        bool threw = false;
        try {
            f.index.get({ LAYER, N_EXPERT });
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        CHECK(f.index.layer_slot(LAYER) == 0);
        CHECK(f.index.layer_slot(LAYER + 1) == -1);

        threw = false;
        try {
            std::vector<moe::ExpertKey> wrong = { { LAYER + 1, 0 } };
            pool->ensure(f.index, store, wrong, ids, st, clock);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        CHECK(threw);
        printf("ok: out-of-range experts and wrong-layer keys are rejected\n");
    }

    // 8. a shared pool holds experts from several layers at once
    {
        fixture f;
        moe::MemoryExpertStore store;
        moe::PoolStats st;
        uint64_t clock = 0;
        std::vector<int32_t> ids;

        // extend the index with a second layer that reuses the same sources (same shapes, as in a real
        // model where every MoE layer has identically shaped expert tensors)
        const uint32_t LAYER_B = LAYER + 2;
        f.index.moe_layers = { LAYER, LAYER_B };
        std::vector<moe::ExpertDescriptor> descs(2 * N_EXPERT);
        for (uint32_t e = 0; e < N_EXPERT; ++e) {
            descs[e] = f.index.descs[e];
            moe::ExpertDescriptor d = f.index.descs[e];
            d.key = { LAYER_B, e };
            descs[N_EXPERT + e] = d;
        }
        f.index.descs = descs;

        moe::LayerPool shared(moe::LayerPool::ANY_LAYER, N_SLOTS,
                              std::vector<ggml_tensor *>{ f.pool[0], f.pool[1] });
        CHECK(shared.is_shared());

        std::vector<moe::ExpertKey> mixed = { { LAYER, 1 }, { LAYER_B, 2 } };
        shared.ensure(f.index, store, mixed, ids, st, clock);
        CHECK(ids[0] != ids[1]);
        f.check_slot_holds(ids[0], 1);
        f.check_slot_holds(ids[1], 2);

        // the same expert id in a different layer is a different key and needs its own slot
        std::vector<moe::ExpertKey> same_id = { { LAYER, 5 }, { LAYER_B, 5 } };
        shared.ensure(f.index, store, same_id, ids, st, clock);
        CHECK(ids[0] != ids[1]);
        printf("ok: a shared pool serves several layers and keys by (layer, expert)\n");
    }

    printf("all moe-pool tests passed\n");
    return 0;
}
