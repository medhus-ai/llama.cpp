#include "expert-runtime.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <cstdio>
#endif

namespace moe {

int ExpertIndex::layer_slot(uint32_t layer) const {
    auto it = std::lower_bound(moe_layers.begin(), moe_layers.end(), layer);
    if (it == moe_layers.end() || *it != layer) {
        return -1;
    }
    return (int) (it - moe_layers.begin());
}

const ExpertDescriptor & ExpertIndex::get(ExpertKey k) const {
    const int ls = layer_slot(k.layer);
    if (ls < 0 || k.expert >= n_expert) {
        throw std::runtime_error("moe: expert key out of range: layer " + std::to_string(k.layer) +
                                 " expert " + std::to_string(k.expert));
    }
    return descs[(size_t) ls * n_expert + k.expert];
}

void MemoryExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    const TensorSlice & s = e.slices.at(slice);
    if (!s.source) {
        throw std::runtime_error("moe: MemoryExpertStore has no resident source for " + s.name);
    }
    if (s.offset + s.bytes > ggml_nbytes(s.source)) {
        throw std::runtime_error("moe: slice out of bounds in " + s.name);
    }
    ggml_backend_tensor_get(s.source, dst, s.offset, s.bytes);
}

BufferedFileExpertStore::BufferedFileExpertStore(const std::string & path) : path_(path) {
#ifndef _WIN32
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("moe: cannot open model file for expert reads: " + path);
    }
#else
    throw std::runtime_error("moe: BufferedFileExpertStore is implemented for POSIX only");
#endif
}

BufferedFileExpertStore::~BufferedFileExpertStore() {
#ifndef _WIN32
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
}

void BufferedFileExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
#ifndef _WIN32
    const TensorSlice & s = e.slices.at(slice);
    if (s.file_offset == 0) {
        throw std::runtime_error("moe: no file offset recorded for " + s.name);
    }
    uint8_t * out = (uint8_t *) dst;
    uint64_t  off = s.file_offset;
    uint64_t  rem = s.bytes;
    while (rem > 0) {
        const ssize_t n = pread(fd_, out, (size_t) rem, (off_t) off);
        if (n <= 0) {
            throw std::runtime_error("moe: short read for " + s.name + " at offset " + std::to_string(off));
        }
        out += n;
        off += (uint64_t) n;
        rem -= (uint64_t) n;
        reads_++;
    }
    bytes_ += s.bytes;
#else
    (void) e; (void) slice; (void) dst;
    throw std::runtime_error("moe: BufferedFileExpertStore is implemented for POSIX only");
#endif
}

VerifyingExpertStore::VerifyingExpertStore(std::unique_ptr<ExpertStore> primary, std::unique_ptr<ExpertStore> reference)
    : primary_(std::move(primary)), reference_(std::move(reference)) {
    name_ = std::string("verify(") + primary_->name() + " vs " + reference_->name() + ")";
}

void VerifyingExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    primary_->read_slice(e, slice, dst);

    const TensorSlice & s = e.slices.at(slice);
    ref_buf_.resize(s.bytes);
    reference_->read_slice(e, slice, ref_buf_.data());

    const uint8_t * a = (const uint8_t *) dst;
    const uint8_t * b = ref_buf_.data();
    for (uint64_t i = 0; i < s.bytes; ++i) {
        if (a[i] != b[i]) {
            throw std::runtime_error("moe: VERIFY FAILED for layer " + std::to_string(e.key.layer) +
                                     " expert " + std::to_string(e.key.expert) + " tensor " + s.name +
                                     " at byte " + std::to_string(i) + ": " + primary_->name() + " gave " +
                                     std::to_string((int) a[i]) + ", " + reference_->name() + " gave " +
                                     std::to_string((int) b[i]));
        }
    }
    checked_++;
}

std::string PoolStats::json() const {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"requests\": %llu, \"hits\": %llu, \"misses\": %llu, \"hit_rate\": %.4f, \"bytes_read\": %llu, "
             "\"evictions\": %llu, \"ubatches\": %llu, \"t_fill_ms\": %.1f}",
             (unsigned long long) requests, (unsigned long long) hits, (unsigned long long) misses,
             requests ? (double) hits / (double) requests : 0.0, (unsigned long long) bytes_read,
             (unsigned long long) evictions, (unsigned long long) ubatches, t_fill_ms);
    return buf;
}

LayerPool::LayerPool(uint32_t layer, uint32_t n_slots, std::vector<ggml_tensor *> pool_tensors)
    : layer_(layer), slots_(n_slots), tensors_(std::move(pool_tensors)) {
    for (auto * t : tensors_) {
        if (t->ne[2] != (int64_t) n_slots) {
            throw std::runtime_error("moe: pool tensor ne[2] != n_slots");
        }
    }
}

int LayerPool::find(ExpertKey k) const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (s.state == SlotState::READY || s.state == SlotState::IN_USE) {
            if (s.key == k) {
                return (int) i;
            }
        }
    }
    return -1;
}

int LayerPool::pick_victim() const {
    int best = -1;
    uint64_t best_use = UINT64_MAX;
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot & s = slots_[i];
        if (s.state == SlotState::FREE) {
            return (int) i;
        }
        if (s.state == SlotState::READY && s.last_use < best_use) {
            best_use = s.last_use;
            best = (int) i;
        }
    }
    return best;  // -1 if everything is IN_USE or LOADING
}

void LayerPool::load(const ExpertIndex & idx, ExpertStore & store, ExpertKey k, int slot, PoolStats & stats) {
    const ExpertDescriptor & d = idx.get(k);
    if (d.slices.size() != tensors_.size()) {
        throw std::runtime_error("moe: descriptor slice count != pool tensor count");
    }
    Slot & s = slots_[slot];
    if (s.state == SlotState::READY) {
        stats.evictions++;
    }
    s.state = SlotState::LOADING;
    s.key = k;
    s.generation++;
    for (size_t i = 0; i < d.slices.size(); ++i) {
        const TensorSlice & sl = d.slices[i];
        ggml_tensor * pt = tensors_[i];
        const uint64_t slot_bytes = pt->nb[2];
        if (sl.bytes != slot_bytes) {
            throw std::runtime_error("moe: slice bytes (" + std::to_string(sl.bytes) + ") != pool slot bytes (" +
                                     std::to_string(slot_bytes) + ") for " + sl.name);
        }
        staging_.resize(sl.bytes);
        store.read_slice(d, i, staging_.data());
        ggml_backend_tensor_set(pt, staging_.data(), (size_t) slot * slot_bytes, sl.bytes);
        stats.bytes_read += sl.bytes;
    }
    s.state = SlotState::READY;
}

void LayerPool::ensure(const ExpertIndex & idx, ExpertStore & store, const std::vector<ExpertKey> & keys,
                       std::vector<int32_t> & slot_ids, PoolStats & stats, uint64_t & clock) {
    const auto t0 = std::chrono::steady_clock::now();
    stats.ubatches++;
    slot_ids.assign(keys.size(), -1);

    // distinct keys, preserving first-seen order
    std::vector<ExpertKey> distinct;
    distinct.reserve(keys.size());
    for (const auto & k : keys) {
        if (k.layer != layer_) {
            throw std::runtime_error("moe: key for layer " + std::to_string(k.layer) + " sent to pool of layer " + std::to_string(layer_));
        }
        if (std::find(distinct.begin(), distinct.end(), k) == distinct.end()) {
            distinct.push_back(k);
        }
    }
    if (distinct.size() > slots_.size()) {
        fprintf(stderr, "moe: layer %u needs %zu distinct experts in one ubatch but the pool has %zu slots; "
                        "raise --moe-pool-slots or lower -ub\n", layer_, distinct.size(), slots_.size());
        throw std::runtime_error("moe: expert working set exceeds pool size");
    }

    // pass 1: pin hits so they cannot be chosen as victims
    for (const auto & k : distinct) {
        const int s = find(k);
        if (s >= 0) {
            slots_[s].state = SlotState::IN_USE;
            slots_[s].last_use = ++clock;
        }
    }
    // pass 2: load misses
    size_t n_miss = 0;
    for (const auto & k : distinct) {
        if (find(k) >= 0) {
            continue;
        }
        const int v = pick_victim();
        if (v < 0) {
            throw std::runtime_error("moe: no evictable slot (all IN_USE/LOADING)");
        }
        load(idx, store, k, v, stats);
        slots_[v].state = SlotState::IN_USE;
        slots_[v].last_use = ++clock;
        n_miss++;
    }
    // resolve ids and release
    for (size_t i = 0; i < keys.size(); ++i) {
        const int s = find(keys[i]);
        if (s < 0) {
            throw std::runtime_error("moe: internal error: key not resident after ensure()");
        }
        slot_ids[i] = s;
    }
    for (auto & s : slots_) {
        if (s.state == SlotState::IN_USE) {
            s.state = SlotState::READY;
        }
    }
    stats.requests += distinct.size();
    stats.misses   += n_miss;
    stats.hits     += distinct.size() - n_miss;
    stats.t_fill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace moe
