#pragma once

// moe-stream-lab: generic hierarchical expert runtime (model-agnostic).
//
// Nothing in this directory may include a model-specific header. Model adapters (src/moe-adapters/)
// translate a loaded model into ExpertDescriptors and hand them to this runtime.

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <string>
#include <vector>

namespace moe {

struct ExpertKey {
    uint32_t layer;
    uint32_t expert;
    bool operator==(const ExpertKey & o) const { return layer == o.layer && expert == o.expert; }
};

// One contiguous slice of bytes that belongs to an expert, inside a named source tensor.
struct TensorSlice {
    std::string   name;         // source tensor name (e.g. blk.3.ffn_up_exps.weight)
    ggml_tensor * source = nullptr; // resident source tensor (MemoryExpertStore) or nullptr
    uint64_t      file_offset = 0;  // absolute offset in the model file (file stores)
    uint64_t      offset = 0;       // byte offset of this expert inside the source tensor
    uint64_t      bytes  = 0;       // bytes of this expert's slice
    ggml_type     type   = GGML_TYPE_COUNT;
};

// All bytes needed to execute one expert. Model-independent: any number of slices (gate/up/down/...).
struct ExpertDescriptor {
    ExpertKey key;
    std::vector<TensorSlice> slices;
    uint64_t total_bytes = 0;
};

// (layer, expert) -> descriptor. Built once at load by an adapter.
struct ExpertIndex {
    uint32_t n_expert = 0;                 // experts per MoE layer
    std::vector<uint32_t> moe_layers;      // layer ids that have routed experts, ascending
    std::vector<ExpertDescriptor> descs;   // indexed by slot_of(layer) * n_expert + expert

    int layer_slot(uint32_t layer) const;  // position of layer in moe_layers, or -1
    const ExpertDescriptor & get(ExpertKey k) const;
    size_t n_layers() const { return moe_layers.size(); }
};

// Source of expert bytes.
class ExpertStore {
public:
    virtual ~ExpertStore() = default;
    // Read the i-th slice of `e` into `dst` (exactly slice.bytes bytes). Must abort/throw on failure,
    // never return partial or substituted data.
    virtual void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) = 0;
    virtual const char * name() const = 0;
};

// Reads from resident ggml tensors (works for any backend via ggml_backend_tensor_get).
class MemoryExpertStore : public ExpertStore {
public:
    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return "memory"; }
};

// Reads expert bytes straight from the model file with positional reads (no mmap, no page-cache hints).
// Requires TensorSlice::file_offset to be the absolute offset of the expert's bytes in the file.
class BufferedFileExpertStore : public ExpertStore {
public:
    explicit BufferedFileExpertStore(const std::string & path);
    ~BufferedFileExpertStore() override;

    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return "file"; }

    uint64_t reads() const { return reads_.load(); }
    uint64_t bytes() const { return bytes_.load(); }

private:
    std::string path_;
    int         fd_ = -1;
    std::atomic<uint64_t> reads_{0};
    std::atomic<uint64_t> bytes_{0};
};

// Reads expert bytes with O_DIRECT, bypassing the page cache. GGUF expert slabs are not block-aligned,
// so each read covers the aligned range containing the slice and the wanted bytes are copied out of it
// (see moe-stream-lab/docs/decisions/0004-direct-io-before-moepack.md). Falls back to buffered reads,
// with an explicit warning, when O_DIRECT is not available on the filesystem.
class DirectIOExpertStore : public ExpertStore {
public:
    explicit DirectIOExpertStore(const std::string & path);
    ~DirectIOExpertStore() override;

    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return direct_ ? "direct" : "direct(fallback:buffered)"; }

    // Read [file_offset, file_offset+bytes) into dst through the aligned staging buffer (thread-safe).
    void read_range(uint64_t file_offset, uint64_t bytes, void * dst, const char * what);

    uint64_t reads() const { return reads_.load(); }
    uint64_t bytes_read() const { return bytes_read_.load(); }   // bytes actually pulled from the device
    uint64_t bytes_used() const { return bytes_used_.load(); }   // bytes the caller asked for
    size_t   align() const { return align_; }
    bool     is_direct() const { return direct_; }

private:
    // staging buffers are per thread so several fetch workers can read concurrently through one store
    struct staging {
        uint8_t * buf = nullptr;
        size_t    cap = 0;
        ~staging();
        uint8_t * get(size_t bytes, size_t align);
    };
    static staging & tls_staging();

    std::string path_;
    int         fd_      = -1;
    bool        direct_  = false;
    size_t      align_   = 4096;
    std::atomic<uint64_t> reads_{0};
    std::atomic<uint64_t> bytes_read_{0};
    std::atomic<uint64_t> bytes_used_{0};
};

// Reads expert bytes from a moepack sidecar (tools/moe-pack): every expert is one contiguous, aligned
// bundle, so a whole expert is fetched with a single O_DIRECT read instead of one unaligned read per
// tensor. TensorSlice::file_offset must point into the pack (set by the adapter from the .moeidx).
// Implemented on top of DirectIOExpertStore's aligned cover-range reader; read_bundle() fetches all
// slices of an expert at once when they are adjacent in the pack.
class PackExpertStore : public ExpertStore {
public:
    explicit PackExpertStore(const std::string & pack_path, bool direct = true);

    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return direct_ ? "pack(direct)" : "pack(buffered)"; }

    // Whole-bundle fetch: slices must be adjacent (slice i+1 offset == slice i offset + bytes).
    // dsts[i] receives slice i. Falls back to per-slice reads if not adjacent.
    void read_bundle(const ExpertDescriptor & e, const std::vector<void *> & dsts);

    uint64_t reads() const { return inner_->reads(); }
    uint64_t bytes_read() const { return inner_->bytes_read(); }
    uint64_t bytes_used() const { return inner_->bytes_used(); }

private:
    std::unique_ptr<DirectIOExpertStore> inner_;
    bool direct_ = true;
};

// Reads every slice through both stores and aborts on the first differing byte. Used by --moe-verify to
// prove that a new storage path returns exactly the bytes the reference path returns.
class VerifyingExpertStore : public ExpertStore {
public:
    VerifyingExpertStore(std::unique_ptr<ExpertStore> primary, std::unique_ptr<ExpertStore> reference);

    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return name_.c_str(); }

    uint64_t checked() const { return checked_.load(); }

private:
    // no shared scratch state: read_slice may be called from several fetch workers at once
    std::unique_ptr<ExpertStore> primary_;
    std::unique_ptr<ExpertStore> reference_;
    std::string                  name_;
    std::atomic<uint64_t>        checked_{0};
};

enum class SlotState : uint8_t { FREE, LOADING, READY, IN_USE };

struct Slot {
    SlotState state = SlotState::FREE;
    ExpertKey key{UINT32_MAX, UINT32_MAX};
    uint64_t  generation = 0;   // bumped on every (re)load; stale lookups can never match
    uint64_t  last_use   = 0;   // logical clock for LRU
};

struct PoolStats {
    uint64_t requests   = 0;   // expert requests (per token per layer per expert)
    uint64_t hits       = 0;
    uint64_t misses     = 0;
    uint64_t bytes_read = 0;   // bytes copied from the store
    uint64_t evictions  = 0;
    uint64_t ubatches   = 0;
    double   t_fill_ms  = 0;   // wall time spent in ensure()
    std::string json() const;
};

// A compact pool of `n_slots`, backed by pool tensors (one per slice kind) whose ne[2] == n_slots.
// The pool does not know what the tensors mean; it only copies bytes.
//
// `layer == ANY_LAYER` makes the pool shared by every MoE layer, which needs all of them to have
// identically shaped and typed expert tensors. A shared pool uses capacity far better than per-layer
// pools (see moe-stream-lab/docs/milestones/M2.md): at a 1 GB budget the simulator measured 34.4% hits
// shared against 3.1% split per layer.
class LayerPool {
public:
    static constexpr uint32_t ANY_LAYER = UINT32_MAX;

    LayerPool(uint32_t layer, uint32_t n_slots, std::vector<ggml_tensor *> pool_tensors);

    bool is_shared() const { return layer_ == ANY_LAYER; }

    // Make every key in `keys` resident; returns slot ids in the same order. Aborts if the distinct set
    // exceeds n_slots (no silent fallback). All keys must belong to this layer.
    void ensure(const ExpertIndex & idx, ExpertStore & store, const std::vector<ExpertKey> & keys,
                std::vector<int32_t> & slot_ids, PoolStats & stats, uint64_t & clock);

    // Number of worker threads used to fetch the misses of one ubatch. 1 keeps the original fully
    // sequential path, so the effect of parallel fetch can be measured by turning it off.
    void set_io_threads(uint32_t n) { io_threads_ = n < 1 ? 1 : n; }
    void set_host_pool(bool host) { host_pool_ = host; }
    uint32_t io_threads() const { return io_threads_; }

    uint32_t n_slots() const { return (uint32_t) slots_.size(); }
    const std::vector<Slot> & slots() const { return slots_; }

private:
    int find(ExpertKey k) const;
    int pick_victim() const;
    void load(const ExpertIndex & idx, ExpertStore & store, ExpertKey k, int slot, PoolStats & stats);
    void fetch_into(ExpertStore & store, const ExpertDescriptor & d, int slot, std::vector<uint8_t> & staging);
    void publish(const ExpertDescriptor & d, int slot, const std::vector<uint8_t> & staging);
    // fetch several (key, slot) pairs, using io_threads_ workers; byte counts are accumulated per worker
    void load_many(const ExpertIndex & idx, ExpertStore & store,
                   const std::vector<std::pair<ExpertKey, int>> & work, PoolStats & stats);

    uint32_t layer_;
    uint32_t io_threads_ = 1;
    bool     host_pool_  = true;   // false when the pool tensors live in device memory
    std::vector<Slot> slots_;
    std::vector<ggml_tensor *> tensors_;  // pool tensor per slice kind, same order as descriptor slices
    std::vector<uint8_t> staging_;
};

} // namespace moe
