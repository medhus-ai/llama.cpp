#include "expert-runtime.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
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
        reads_.fetch_add(1, std::memory_order_relaxed);
    }
    bytes_.fetch_add(s.bytes, std::memory_order_relaxed);
#else
    (void) e; (void) slice; (void) dst;
    throw std::runtime_error("moe: BufferedFileExpertStore is implemented for POSIX only");
#endif
}

#ifndef _WIN32
// Logical block size of the device holding `path`, or 0 if it cannot be determined.
static size_t device_logical_block_size(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    char sysfs[256];
    snprintf(sysfs, sizeof(sysfs), "/sys/dev/block/%u:%u/queue/logical_block_size",
             (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
    FILE * f = fopen(sysfs, "r");
    if (!f) {
        // partitions expose the queue through their parent
        snprintf(sysfs, sizeof(sysfs), "/sys/dev/block/%u:%u/../queue/logical_block_size",
                 (unsigned) major(st.st_dev), (unsigned) minor(st.st_dev));
        f = fopen(sysfs, "r");
    }
    if (!f) {
        return 0;
    }
    unsigned v = 0;
    const int n = fscanf(f, "%u", &v);
    fclose(f);
    return n == 1 ? (size_t) v : 0;
}
#endif

DirectIOExpertStore::DirectIOExpertStore(const std::string & path) : path_(path) {
#ifndef _WIN32
    const size_t bs = device_logical_block_size(path);
    align_ = bs ? bs : 4096;

    fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ >= 0) {
        direct_ = true;
    } else {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("moe: cannot open model file for expert reads: " + path);
        }
        direct_ = false;
        fprintf(stderr, "moe: O_DIRECT is not available for %s, falling back to buffered reads "
                        "(measurements will include page-cache effects)\n", path.c_str());
    }
    fprintf(stderr, "moe: direct store on %s, alignment %zu bytes, O_DIRECT %s\n",
            path.c_str(), align_, direct_ ? "on" : "off");
#else
    throw std::runtime_error("moe: DirectIOExpertStore is implemented for POSIX only");
#endif
}

DirectIOExpertStore::~DirectIOExpertStore() {
#ifndef _WIN32
    if (fd_ >= 0) {
        close(fd_);
    }
#endif
}

DirectIOExpertStore::staging::~staging() {
#ifndef _WIN32
    if (buf) {
        free(buf);
    }
#endif
}

uint8_t * DirectIOExpertStore::staging::get(size_t bytes, size_t align) {
#ifndef _WIN32
    if (cap >= bytes) {
        return buf;
    }
    if (buf) {
        free(buf);
        buf = nullptr;
        cap = 0;
    }
    void * p = nullptr;
    if (posix_memalign(&p, align, bytes) != 0 || !p) {
        throw std::runtime_error("moe: cannot allocate an aligned staging buffer of " + std::to_string(bytes) + " bytes");
    }
    buf = (uint8_t *) p;
    cap = bytes;
    return buf;
#else
    (void) bytes; (void) align;
    return nullptr;
#endif
}

DirectIOExpertStore::staging & DirectIOExpertStore::tls_staging() {
    thread_local staging st;
    return st;
}

void DirectIOExpertStore::read_range(uint64_t file_offset, uint64_t bytes, void * dst, const char * what) {
#ifndef _WIN32
    const uint64_t begin = (file_offset / align_) * align_;
    const uint64_t end   = ((file_offset + bytes + align_ - 1) / align_) * align_;
    const size_t   span  = (size_t) (end - begin);
    uint8_t * buf = tls_staging().get(span, align_);

    size_t   got = 0;
    uint64_t n_reads = 0;
    while (got < span) {
        const ssize_t n = pread(fd_, buf + got, span - got, (off_t) (begin + got));
        if (n < 0) {
            throw std::runtime_error(std::string("moe: read error for ") + what + " at offset " + std::to_string(begin + got));
        }
        if (n == 0) {
            if (begin + got < file_offset + bytes) {
                throw std::runtime_error(std::string("moe: unexpected EOF reading ") + what);
            }
            break;
        }
        got += (size_t) n;
        n_reads++;
    }
    if (begin + got < file_offset + bytes) {
        throw std::runtime_error(std::string("moe: short read for ") + what);
    }
    memcpy(dst, buf + (file_offset - begin), bytes);
    reads_.fetch_add(n_reads, std::memory_order_relaxed);
    bytes_read_.fetch_add(got, std::memory_order_relaxed);
    bytes_used_.fetch_add(bytes, std::memory_order_relaxed);
#else
    (void) file_offset; (void) bytes; (void) dst; (void) what;
    throw std::runtime_error("moe: DirectIOExpertStore is implemented for POSIX only");
#endif
}

void DirectIOExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    const TensorSlice & s = e.slices.at(slice);
    if (s.file_offset == 0) {
        throw std::runtime_error("moe: no file offset recorded for " + s.name);
    }
    read_range(s.file_offset, s.bytes, dst, s.name.c_str());
}

PackExpertStore::PackExpertStore(const std::string & pack_path, bool direct) : direct_(direct) {
    inner_ = std::make_unique<DirectIOExpertStore>(pack_path);
    if (direct && !inner_->is_direct()) {
        direct_ = false;
    }
}

void PackExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    inner_->read_slice(e, slice, dst);
}

void PackExpertStore::read_bundle(const ExpertDescriptor & e, const std::vector<void *> & dsts) {
    if (dsts.size() != e.slices.size()) {
        throw std::runtime_error("moe: read_bundle: destination count != slice count");
    }
    bool adjacent = true;
    for (size_t i = 1; i < e.slices.size(); ++i) {
        if (e.slices[i].file_offset != e.slices[i - 1].file_offset + e.slices[i - 1].bytes) {
            adjacent = false;
            break;
        }
    }
    if (!adjacent || e.slices.empty()) {
        for (size_t i = 0; i < e.slices.size(); ++i) {
            inner_->read_slice(e, i, dsts[i]);
        }
        return;
    }
    // one read for the whole bundle, then scatter into the per-tensor destinations
    std::vector<uint8_t> & tmp = [] () -> std::vector<uint8_t> & { thread_local std::vector<uint8_t> t; return t; }();
    tmp.resize(e.total_bytes);
    inner_->read_range(e.slices[0].file_offset, e.total_bytes, tmp.data(), "expert bundle");
    uint64_t off = 0;
    for (size_t i = 0; i < e.slices.size(); ++i) {
        memcpy(dsts[i], tmp.data() + off, e.slices[i].bytes);
        off += e.slices[i].bytes;
    }
}

CachingExpertStore::CachingExpertStore(std::unique_ptr<ExpertStore> inner, uint64_t capacity_bytes, uint64_t bundle_bytes,
                                       ggml_backend_buffer_type_t host_buft)
    : inner_(std::move(inner)), bundle_bytes_(bundle_bytes) {
    n_slots_ = bundle_bytes ? capacity_bytes / bundle_bytes : 0;
    if (n_slots_ > 0) {
        const size_t bytes = (size_t) (n_slots_ * bundle_bytes_);
        if (host_buft) {
            arena_buf_ = ggml_backend_buft_alloc_buffer(host_buft, bytes);
            if (!arena_buf_) {
                fprintf(stderr, "moe: could not allocate a %zu MiB pinned arena from %s, using pageable memory\n",
                        bytes >> 20, ggml_backend_buft_name(host_buft));
            } else {
                arena_ = (uint8_t *) ggml_backend_buffer_get_base(arena_buf_);
            }
        }
        if (!arena_) {
            void * p = nullptr;
            if (posix_memalign(&p, 4096, bytes) != 0 || !p) {
                throw std::runtime_error("moe: cannot allocate the host expert cache arena");
            }
            arena_ = (uint8_t *) p;
        }
        slabs_.resize((size_t) n_slots_);
    }
    name_ = std::string(arena_buf_ ? "hostcache-pinned(" : "hostcache(") + inner_->name() + ")";
}

CachingExpertStore::~CachingExpertStore() {
    if (arena_buf_) {
        ggml_backend_buffer_free(arena_buf_);
    } else if (arena_) {
        free(arena_);
    }
}

void CachingExpertStore::lru_unlink_locked(int i) {
    Slab & s = slabs_[i];
    if (s.prev >= 0) { slabs_[s.prev].next = s.next; } else if (head_ == i) { head_ = s.next; }
    if (s.next >= 0) { slabs_[s.next].prev = s.prev; } else if (tail_ == i) { tail_ = s.prev; }
    s.prev = s.next = -1;
}

void CachingExpertStore::lru_touch_locked(int i) {
    lru_unlink_locked(i);
    Slab & s = slabs_[i];
    s.next = head_;
    if (head_ >= 0) { slabs_[head_].prev = i; }
    head_ = i;
    if (tail_ < 0) { tail_ = i; }
}

int CachingExpertStore::lru_victim_locked() const {
    for (int i = tail_; i >= 0; i = slabs_[i].prev) {
        if (slabs_[i].state == St::READY && slabs_[i].readers == 0) {
            return i;
        }
    }
    return -1;
}

int CachingExpertStore::acquire_locked(std::unique_lock<std::mutex> & lock, ExpertKey key, int & reserved) {
    reserved = -1;
    for (;;) {
        auto it = index_.find({ key.layer, key.expert });
        if (it != index_.end()) {
            Slab & s = slabs_[it->second];
            if (s.state == St::READY) {
                return it->second;
            }
            cv_.wait(lock);   // another thread is loading it
            continue;
        }
        // miss: reserve a slab (a FREE one first, else the LRU READY one not being read)
        int v = -1;
        if (n_ready_ + 0 < n_slots_) {
            for (size_t i = 0; i < slabs_.size(); ++i) {
                if (slabs_[i].state == St::FREE) { v = (int) i; break; }
            }
        }
        if (v < 0) {
            v = lru_victim_locked();
        }
        if (v < 0) {
            cv_.wait(lock);   // everything is LOADING or being read; wait for something to settle
            continue;
        }
        Slab & s = slabs_[v];
        if (s.state == St::READY) {
            index_.erase({ s.layer, s.expert });
            n_ready_--;
            stats_.evictions++;
        }
        lru_unlink_locked(v);
        s.state = St::LOADING;
        s.layer = key.layer;
        s.expert = key.expert;
        index_[{ key.layer, key.expert }] = v;
        reserved = v;
        return -1;
    }
}

int CachingExpertStore::resident_index(const ExpertDescriptor & e) {
    std::unique_lock<std::mutex> lock(mtx_);
    int reserved = -1;
    int i = acquire_locked(lock, e.key, reserved);
    if (i < 0) {
        // miss: read from storage straight into the reserved slab, outside the lock
        stats_.misses++;
        lock.unlock();
        uint8_t * slab = arena_ + (size_t) reserved * bundle_bytes_;
        try {
            uint64_t off = 0;
            for (size_t k = 0; k < e.slices.size(); ++k) {
                inner_->read_slice(e, k, slab + off);
                off += e.slices[k].bytes;
            }
        } catch (...) {
            lock.lock();
            index_.erase({ e.key.layer, e.key.expert });
            slabs_[reserved].state = St::FREE;
            cv_.notify_all();
            throw;
        }
        lock.lock();
        slabs_[reserved].state = St::READY;
        n_ready_++;
        stats_.bytes_inserted += bundle_bytes_;
        i = reserved;
        cv_.notify_all();
    } else {
        stats_.hits++;
        stats_.bytes_from_cache += bundle_bytes_;
    }
    lru_touch_locked(i);
    slabs_[i].readers++;
    return i;
}

const uint8_t * CachingExpertStore::acquire_bundle(const ExpertDescriptor & e) {
    if (n_slots_ == 0 || e.total_bytes != bundle_bytes_) {
        throw std::runtime_error("moe: acquire_bundle on a pass-through host cache");
    }
    const int i = resident_index(e);
    return arena_ + (size_t) i * bundle_bytes_;
}

void CachingExpertStore::release_bundle(const ExpertDescriptor & e) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = index_.find({ e.key.layer, e.key.expert });
    if (it != index_.end() && slabs_[it->second].readers > 0) {
        slabs_[it->second].readers--;
    }
    cv_.notify_all();
}

void CachingExpertStore::read_bundle(const ExpertDescriptor & e, const std::vector<void *> & dsts) {
    if (n_slots_ == 0 || e.total_bytes != bundle_bytes_) {
        // pass-through (cache disabled or a descriptor of a different size)
        for (size_t i = 0; i < e.slices.size(); ++i) {
            inner_->read_slice(e, i, dsts[i]);
        }
        return;
    }
    const int i = resident_index(e);
    // copy out without holding the lock; the slab cannot be evicted while readers > 0
    const uint8_t * slab = arena_ + (size_t) i * bundle_bytes_;
    uint64_t off = 0;
    for (size_t k = 0; k < e.slices.size(); ++k) {
        memcpy(dsts[k], slab + off, e.slices[k].bytes);
        off += e.slices[k].bytes;
    }
    release_bundle(e);
}

void CachingExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    // single-slice reads (verify wrapper) go through the bundle path with scratch for the other slices
    std::vector<std::vector<uint8_t>> scratch(e.slices.size());
    std::vector<void *> dsts(e.slices.size());
    for (size_t k = 0; k < e.slices.size(); ++k) {
        if (k == slice) {
            dsts[k] = dst;
        } else {
            scratch[k].resize(e.slices[k].bytes);
            dsts[k] = scratch[k].data();
        }
    }
    read_bundle(e, dsts);
}

CachingExpertStore::Stats CachingExpertStore::stats() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return stats_;
}

uint64_t CachingExpertStore::resident_bytes() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return n_ready_ * bundle_bytes_;
}

VerifyingExpertStore::VerifyingExpertStore(std::unique_ptr<ExpertStore> primary, std::unique_ptr<ExpertStore> reference)
    : primary_(std::move(primary)), reference_(std::move(reference)) {
    name_ = std::string("verify(") + primary_->name() + " vs " + reference_->name() + ")";
}

void VerifyingExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    primary_->read_slice(e, slice, dst);

    const TensorSlice & s = e.slices.at(slice);
    std::vector<uint8_t> ref_buf(s.bytes);   // per call: this may run on several fetch workers at once
    reference_->read_slice(e, slice, ref_buf.data());

    const uint8_t * a = (const uint8_t *) dst;
    const uint8_t * b = ref_buf.data();
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
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            host_pool_ = false;
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

FetchWorkers::FetchWorkers(uint32_t n_threads) {
    for (uint32_t i = 0; i < n_threads; ++i) {
        threads_.emplace_back([this]() { loop(); });
    }
}

FetchWorkers::~FetchWorkers() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto & t : threads_) {
        t.join();
    }
}

void FetchWorkers::loop() {
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return stop_ || gen_ != seen; });
        if (stop_) {
            return;
        }
        seen = gen_;
        for (;;) {
            if (next_ >= n_) {
                break;
            }
            const size_t i = next_++;
            lock.unlock();
            try {
                (*fn_)(i);
            } catch (...) {
                lock.lock();
                errors_.push_back(std::current_exception());
                lock.unlock();
            }
            lock.lock();
            done_++;
        }
        if (done_ == n_) {
            done_cv_.notify_all();
        }
    }
}

void FetchWorkers::run(size_t n, const std::function<void(size_t)> & fn) {
    std::unique_lock<std::mutex> lock(mtx_);
    fn_ = &fn; n_ = n; next_ = 0; done_ = 0; errors_.clear(); gen_++;
    cv_.notify_all();
    done_cv_.wait(lock, [&] { return done_ == n_; });
    fn_ = nullptr;
    if (!errors_.empty()) {
        std::exception_ptr e = errors_.front();
        errors_.clear();
        std::rethrow_exception(e);
    }
}

void LayerPool::set_io_threads(uint32_t n) {
    io_threads_ = n < 1 ? 1 : n;
    workers_.reset();
    if (io_threads_ > 1) {
        workers_ = std::make_unique<FetchWorkers>(io_threads_);
    }
}

// Read every slice of `d` into `staging` (contiguous, slice order). Safe on any thread.
static void read_expert(ExpertStore & store, const ExpertDescriptor & d, std::vector<uint8_t> & staging) {
    staging.resize(d.total_bytes);
    std::vector<void *> dsts(d.slices.size());
    uint64_t off = 0;
    for (size_t i = 0; i < d.slices.size(); ++i) {
        dsts[i] = staging.data() + off;
        off += d.slices[i].bytes;
    }
    if (auto * pack = dynamic_cast<PackExpertStore *>(&store)) {
        pack->read_bundle(d, dsts);
    } else if (auto * cache = dynamic_cast<CachingExpertStore *>(&store)) {
        cache->read_bundle(d, dsts);
    } else {
        for (size_t i = 0; i < d.slices.size(); ++i) {
            store.read_slice(d, i, dsts[i]);
        }
    }
}

// Copy a staged expert into `slot` of the pool tensors (host memcpy or H2D, depending on the buffer).
void LayerPool::publish(const ExpertDescriptor & d, int slot, const std::vector<uint8_t> & staging) {
    uint64_t off = 0;
    for (size_t i = 0; i < d.slices.size(); ++i) {
        ggml_backend_tensor_set(tensors_[i], staging.data() + off, (size_t) slot * tensors_[i]->nb[2], d.slices[i].bytes);
        off += d.slices[i].bytes;
    }
}

void LayerPool::fetch_into(ExpertStore & store, const ExpertDescriptor & d, int slot, std::vector<uint8_t> & staging) {
    read_expert(store, d, staging);
    publish(d, slot, staging);
}

void LayerPool::load(const ExpertIndex & idx, ExpertStore & store, ExpertKey k, int slot, PoolStats & stats) {
    const ExpertDescriptor & d = idx.get(k);
    if (d.slices.size() != tensors_.size()) {
        throw std::runtime_error("moe: descriptor slice count != pool tensor count");
    }
    Slot & s = slots_[slot];
    s.state = SlotState::LOADING;
    s.key = k;
    s.generation++;
    for (size_t i = 0; i < d.slices.size(); ++i) {
        const TensorSlice & sl = d.slices[i];
        const uint64_t slot_bytes = tensors_[i]->nb[2];
        if (sl.bytes != slot_bytes) {
            throw std::runtime_error("moe: slice bytes (" + std::to_string(sl.bytes) + ") != pool slot bytes (" +
                                     std::to_string(slot_bytes) + ") for " + sl.name);
        }
    }
    fetch_into(store, d, slot, staging_);
    stats.bytes_read += d.total_bytes;
    s.state = SlotState::READY;
}

void LayerPool::load_many(const ExpertIndex & idx, ExpertStore & store,
                          const std::vector<std::pair<ExpertKey, int>> & work, PoolStats & stats) {
    if (work.empty()) {
        return;
    }
    const uint32_t n_threads = std::min<uint32_t>(io_threads_, (uint32_t) work.size());

    if (n_threads <= 1) {
        for (const auto & w : work) {
            load(idx, store, w.first, w.second, stats);
        }
        return;
    }

    // Workers only move bytes; slot bookkeeping stays on this thread (before and after), so there is no
    // shared mutable state except the per-worker staging buffers inside the store.
    for (const auto & w : work) {
        Slot & sl = slots_[w.second];
        sl.state = SlotState::LOADING;
        sl.key   = w.first;
        sl.generation++;
    }

    std::atomic<uint64_t> bytes{0};
    std::vector<std::vector<uint8_t>> staged(host_pool_ ? 0 : work.size());
    auto * cache = host_pool_ ? nullptr : dynamic_cast<CachingExpertStore *>(&store);
    std::vector<const uint8_t *> views(cache ? work.size() : 0, nullptr);
    // one staging buffer per worker for host pools (workers write straight into the pool tensors)
    std::vector<std::vector<uint8_t>> worker_buf(workers_ ? workers_->size() : 1);
    std::atomic<uint32_t> buf_ticket{0};
    thread_local int my_buf = -1;   // index into worker_buf, assigned on first use per worker thread

    auto task = [&](size_t i) {
        const ExpertDescriptor & d = idx.get(work[i].first);
        const int slot = work[i].second;
        if (d.slices.size() != tensors_.size()) {
            throw std::runtime_error("moe: descriptor slice count != pool tensor count");
        }
        for (size_t k = 0; k < d.slices.size(); ++k) {
            if (d.slices[k].bytes != (uint64_t) tensors_[k]->nb[2]) {
                throw std::runtime_error("moe: slice bytes != pool slot bytes for " + d.slices[k].name);
            }
        }
        if (host_pool_) {
            if (my_buf < 0 || (size_t) my_buf >= worker_buf.size()) {
                my_buf = (int) (buf_ticket.fetch_add(1) % worker_buf.size());
            }
            fetch_into(store, d, slot, worker_buf[my_buf]);
        } else if (cache) {
            // device pool + host tier: make the bundle resident in the (pinned) arena; the H2D copy below
            // reads straight from it, so there is no intermediate pageable copy
            views[i] = cache->acquire_bundle(d);
        } else {
            read_expert(store, d, staged[i]);
        }
        bytes.fetch_add(d.total_bytes, std::memory_order_relaxed);
    };
    // no partial state is published on error: slots stay LOADING and the exception propagates
    workers_->run(work.size(), task);

    if (!host_pool_) {
        for (size_t i = 0; i < work.size(); ++i) {
            const ExpertDescriptor & d = idx.get(work[i].first);
            if (cache) {
                uint64_t off = 0;
                for (size_t k = 0; k < d.slices.size(); ++k) {
                    ggml_backend_tensor_set(tensors_[k], views[i] + off, (size_t) work[i].second * tensors_[k]->nb[2], d.slices[k].bytes);
                    off += d.slices[k].bytes;
                }
                cache->release_bundle(d);
            } else {
                publish(d, work[i].second, staged[i]);
            }
        }
    }
    for (const auto & w : work) {
        slots_[w.second].state = SlotState::READY;
    }
    stats.bytes_read += bytes.load();
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
        if (layer_ != ANY_LAYER && k.layer != layer_) {
            throw std::runtime_error("moe: key for layer " + std::to_string(k.layer) + " sent to pool of layer " + std::to_string(layer_));
        }
        if (std::find(distinct.begin(), distinct.end(), k) == distinct.end()) {
            distinct.push_back(k);
        }
    }
    if (distinct.size() > slots_.size()) {
        const std::string who = layer_ == ANY_LAYER ? std::string("the shared expert pool")
                                                    : ("layer " + std::to_string(layer_));
        fprintf(stderr, "moe: %s needs %zu distinct experts in one ubatch but the pool has %zu slots; "
                        "raise --moe-pool-slots or lower -ub\n", who.c_str(), distinct.size(), slots_.size());
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
    // pass 2: choose a slot for every miss first, then fetch them together
    std::vector<std::pair<ExpertKey, int>> work;
    for (const auto & k : distinct) {
        if (find(k) >= 0) {
            continue;
        }
        const int v = pick_victim();
        if (v < 0) {
            throw std::runtime_error("moe: no evictable slot (all IN_USE/LOADING)");
        }
        // reserve the slot immediately so the next iteration cannot pick it again; this is the eviction
        if (slots_[v].state == SlotState::READY) {
            stats.evictions++;
        }
        slots_[v].state = SlotState::LOADING;
        slots_[v].key   = k;
        work.emplace_back(k, v);
    }
    const size_t n_miss = work.size();
    load_many(idx, store, work, stats);
    for (const auto & w : work) {
        slots_[w.second].state = SlotState::IN_USE;
        slots_[w.second].last_use = ++clock;
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
