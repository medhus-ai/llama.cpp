#pragma once

// moe-stream-lab: generic hierarchical expert runtime (model-agnostic).
//
// Nothing in this directory may include a model-specific header. Model adapters (src/moe-adapters/)
// translate a loaded model into ExpertDescriptors and hand them to this runtime.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
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

    uint64_t reads() const { return reads_; }
    uint64_t bytes() const { return bytes_; }

private:
    std::string path_;
    int         fd_ = -1;
    uint64_t    reads_ = 0;
    uint64_t    bytes_ = 0;
};

// Reads every slice through both stores and aborts on the first differing byte. Used by --moe-verify to
// prove that a new storage path returns exactly the bytes the reference path returns.
class VerifyingExpertStore : public ExpertStore {
public:
    VerifyingExpertStore(std::unique_ptr<ExpertStore> primary, std::unique_ptr<ExpertStore> reference);

    void read_slice(const ExpertDescriptor & e, size_t slice, void * dst) override;
    const char * name() const override { return name_.c_str(); }

    uint64_t checked() const { return checked_; }

private:
    std::unique_ptr<ExpertStore> primary_;
    std::unique_ptr<ExpertStore> reference_;
    std::vector<uint8_t>         ref_buf_;
    std::string                  name_;
    uint64_t                     checked_ = 0;
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

// One compact pool of `n_slots` for one MoE layer, backed by pool tensors (one per slice kind) whose
// ne[2] == n_slots. The pool does not know what the tensors mean; it only copies bytes.
class LayerPool {
public:
    LayerPool(uint32_t layer, uint32_t n_slots, std::vector<ggml_tensor *> pool_tensors);

    // Make every key in `keys` resident; returns slot ids in the same order. Aborts if the distinct set
    // exceeds n_slots (no silent fallback). All keys must belong to this layer.
    void ensure(const ExpertIndex & idx, ExpertStore & store, const std::vector<ExpertKey> & keys,
                std::vector<int32_t> & slot_ids, PoolStats & stats, uint64_t & clock);

    uint32_t n_slots() const { return (uint32_t) slots_.size(); }
    const std::vector<Slot> & slots() const { return slots_; }

private:
    int find(ExpertKey k) const;
    int pick_victim() const;
    void load(const ExpertIndex & idx, ExpertStore & store, ExpertKey k, int slot, PoolStats & stats);

    uint32_t layer_;
    std::vector<Slot> slots_;
    std::vector<ggml_tensor *> tensors_;  // pool tensor per slice kind, same order as descriptor slices
    std::vector<uint8_t> staging_;
};

} // namespace moe
