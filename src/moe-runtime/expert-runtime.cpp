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
#ifdef __linux__
#include <sys/sysmacros.h>
#endif
#include <unistd.h>
#ifdef LLAMA_MOE_IO_URING
#include <liburing.h>
#endif
#else
#include <cstdio>
#endif


// A pool tensor holds one expert per slot. Weight tensors are 3D and index slots on ne[2];
// per-expert vectors (Gemma 4's ffn_down_exps.scale) are 1-D and index slots on ne[0].
static inline uint64_t moe_slot_stride(const ggml_tensor * t) {
    return t->ne[2] > 1 ? t->nb[2] : t->nb[0];
}

static inline int64_t moe_slot_count(const ggml_tensor * t) {
    return t->ne[2] > 1 ? t->ne[2] : t->ne[0];
}

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

#ifndef _WIN32
// Logical block size of the device holding `path`, or 0 if it cannot be determined.
static size_t device_logical_block_size(const std::string & path) {
#ifndef __linux__
    (void) path;
    return 0; // no sysfs: callers use 4096
#else
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
#endif
}

#ifdef __APPLE__
#define MOE_UNCACHED_NAME "F_NOCACHE"
#else
#define MOE_UNCACHED_NAME "O_DIRECT"
#endif

// Open `path` read-only bypassing the page cache (O_DIRECT, or F_NOCACHE on macOS); falls back to a plain
// open with `direct` = false. Returns -1 if the file cannot be opened at all.
static int open_uncached(const std::string & path, bool & direct) {
#ifdef __APPLE__
    const int fd = open(path.c_str(), O_RDONLY);
    direct = fd >= 0 && fcntl(fd, F_NOCACHE, 1) == 0;
    return fd;
#else
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    direct = fd >= 0;
    if (fd < 0) {
        fd = open(path.c_str(), O_RDONLY);
    }
    return fd;
#endif
}
#endif

DirectIOExpertStore::DirectIOExpertStore(const std::vector<std::string> & paths) : paths_(paths) {
#ifndef _WIN32
    if (paths_.empty()) {
        throw std::runtime_error("moe: DirectIOExpertStore needs at least one model path");
    }
    path_ = paths_[0];
    const size_t bs = device_logical_block_size(path_);
    align_ = bs ? bs : 4096;

    fds_.resize(paths_.size(), -1);
    for (size_t i = 0; i < paths_.size(); ++i) {
        bool direct = false;
        fds_[i] = open_uncached(paths_[i], direct);
        if (fds_[i] < 0) {
            throw std::runtime_error("moe: cannot open model file for expert reads: " + paths_[i]);
        }
        if (i == 0) {
            direct_ = direct;
        }
        if (!direct) {
            fprintf(stderr, "moe: " MOE_UNCACHED_NAME " is not available for %s, falling back to buffered reads "
                            "(measurements will include page-cache effects)\n", paths_[i].c_str());
        }
    }
    fd_ = fds_[0];
    const std::string extra = paths_.size() > 1
        ? " (+" + std::to_string(paths_.size() - 1) + " more shards)" : std::string();
    fprintf(stderr, "moe: direct store on %s%s, alignment %zu bytes, " MOE_UNCACHED_NAME " %s\n",
            path_.c_str(), extra.c_str(), align_, direct_ ? "on" : "off");
#else
    throw std::runtime_error("moe: DirectIOExpertStore is implemented for POSIX only");
#endif
}

DirectIOExpertStore::~DirectIOExpertStore() {
#ifndef _WIN32
    for (int fd : fds_) {
        if (fd >= 0) {
            close(fd);
        }
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

void DirectIOExpertStore::read_range(uint64_t file_offset, uint64_t bytes, void * dst, const char * what, uint16_t shard) {
#ifndef _WIN32
    const uint64_t begin = (file_offset / align_) * align_;
    const uint64_t end   = ((file_offset + bytes + align_ - 1) / align_) * align_;
    const size_t   span  = (size_t) (end - begin);
    uint8_t * buf = tls_staging().get(span, align_);
    if (shard >= fds_.size()) {
        throw std::runtime_error(std::string("moe: slice names shard ") + std::to_string(shard) + " but the store has " + std::to_string(fds_.size()));
    }
    const int fd = fds_[shard];

    size_t   got = 0;
    uint64_t n_reads = 0;
    while (got < span) {
        const ssize_t n = pread(fd, buf + got, span - got, (off_t) (begin + got));
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
    (void) file_offset; (void) bytes; (void) dst; (void) what; (void) shard;
    throw std::runtime_error("moe: DirectIOExpertStore is implemented for POSIX only");
#endif
}

void DirectIOExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    const TensorSlice & s = e.slices.at(slice);
    if (s.file_offset == 0) {
        throw std::runtime_error("moe: no file offset recorded for " + s.name);
    }
    read_range(s.file_offset, s.bytes, dst, s.name.c_str(), s.shard);
}

IoUringExpertStore::IoUringExpertStore(const std::vector<std::string> & paths, unsigned queue_depth) : paths_(paths), depth_(queue_depth) {
#ifndef _WIN32
    if (paths_.empty()) {
        throw std::runtime_error("moe: IoUringExpertStore needs at least one model path");
    }
    path_ = paths_[0];
    const size_t bs = device_logical_block_size(path_);
    align_ = bs ? bs : 4096;
    fds_.resize(paths_.size(), -1);
    for (size_t i = 0; i < paths_.size(); ++i) {
        bool direct = false;
        fds_[i] = open_uncached(paths_[i], direct);
        if (fds_[i] < 0) {
            throw std::runtime_error("moe: cannot open model file for expert reads: " + paths_[i]);
        }
        if (i == 0) {
            direct_ = direct;
        }
        if (!direct) {
            fprintf(stderr, "moe: " MOE_UNCACHED_NAME " is not available for %s, io_uring store falls back to buffered reads\n", paths_[i].c_str());
        }
    }
    fd_ = fds_[0];
#ifdef LLAMA_MOE_IO_URING
    auto * ring = new io_uring();
    if (io_uring_queue_init(depth_, ring, 0) == 0) {
        ring_ = ring;
        ring_ok_ = true;
    } else {
        delete ring;
        fprintf(stderr, "moe: io_uring_queue_init failed, io_uring store falls back to pread\n");
    }
#else
    fprintf(stderr, "moe: built without liburing, io_uring store falls back to pread\n");
#endif
    name_ = std::string(ring_ok_ ? "uring" : "uring(fallback:pread)") + (direct_ ? "" : "(buffered)");
    const std::string extra = paths_.size() > 1
        ? " (+" + std::to_string(paths_.size() - 1) + " more shards)" : std::string();
    fprintf(stderr, "moe: io_uring store on %s%s, alignment %zu bytes, " MOE_UNCACHED_NAME " %s, ring %s (depth %u)\n",
            path_.c_str(), extra.c_str(), align_, direct_ ? "on" : "off", ring_ok_ ? "on" : "off", depth_);
#else
    throw std::runtime_error("moe: IoUringExpertStore is implemented for Linux only");
#endif
}

IoUringExpertStore::~IoUringExpertStore() {
#ifndef _WIN32
#ifdef LLAMA_MOE_IO_URING
    if (ring_) {
        io_uring_queue_exit((io_uring *) ring_);
        delete (io_uring *) ring_;
    }
#endif
    if (arena_) {
        free(arena_);
    }
    for (int fd : fds_) {
        if (fd >= 0) {
            close(fd);
        }
    }
#endif
}

void IoUringExpertStore::read_slice(const ExpertDescriptor & e, size_t slice, void * dst) {
    const TensorSlice & s = e.slices.at(slice);
    if (s.file_offset == 0) {
        throw std::runtime_error("moe: no file offset recorded for " + s.name);
    }
    read_batch({ BatchRead{ s.file_offset, s.bytes, dst, s.name.c_str(), s.shard } });
}

void IoUringExpertStore::read_batch(const std::vector<BatchRead> & reqs) {
#ifndef _WIN32
    if (reqs.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx_);

    // cover ranges, laid out back to back in one aligned arena
    struct span_t { uint64_t begin; size_t span; size_t arena_off; };
    std::vector<span_t> spans(reqs.size());
    size_t total = 0;
    for (size_t i = 0; i < reqs.size(); ++i) {
        const uint64_t begin = (reqs[i].file_offset / align_) * align_;
        const uint64_t end   = ((reqs[i].file_offset + reqs[i].bytes + align_ - 1) / align_) * align_;
        spans[i] = { begin, (size_t) (end - begin), total };
        total += (size_t) (end - begin);
    }
    if (arena_cap_ < total) {
        if (arena_) {
            free(arena_);
            arena_ = nullptr;
        }
        void * p = nullptr;
        if (posix_memalign(&p, align_, total) != 0 || !p) {
            throw std::runtime_error("moe: cannot allocate an aligned io_uring arena of " + std::to_string(total) + " bytes");
        }
        arena_ = (uint8_t *) p;
        arena_cap_ = total;
    }

    uint64_t n_reads = 0;
#ifdef LLAMA_MOE_IO_URING
    if (ring_ok_) {
        auto * ring = (io_uring *) ring_;
        // each request may need several SQEs (short reads are completed by re-submitting the remainder)
        std::vector<size_t> got(reqs.size(), 0);
        size_t pending = 0;
        size_t next = 0;
        size_t done = 0;
        auto submit_one = [&](size_t i) {
            io_uring_sqe * sqe = io_uring_get_sqe(ring);
            if (!sqe) {
                return false;
            }
            io_uring_prep_read(sqe, fds_.at(reqs[i].shard), arena_ + spans[i].arena_off + got[i], (unsigned) (spans[i].span - got[i]),
                               (__u64) (spans[i].begin + got[i]));
            io_uring_sqe_set_data64(sqe, (__u64) i);
            pending++;
            return true;
        };
        while (done < reqs.size()) {
            while (next < reqs.size() && pending < depth_ && submit_one(next)) {
                next++;
            }
            const int rc = io_uring_submit_and_wait(ring, 1);
            if (rc < 0) {
                throw std::runtime_error("moe: io_uring_submit_and_wait failed: " + std::to_string(-rc));
            }
            io_uring_cqe * cqe = nullptr;
            unsigned head = 0;
            unsigned seen = 0;
            io_uring_for_each_cqe(ring, head, cqe) {
                seen++;
                pending--;
                const size_t i = (size_t) io_uring_cqe_get_data64(cqe);
                if (cqe->res < 0) {
                    io_uring_cq_advance(ring, seen);
                    throw std::runtime_error(std::string("moe: io_uring read error for ") + reqs[i].what + ": " + std::to_string(-cqe->res));
                }
                n_reads++;
                got[i] += (size_t) cqe->res;
                if (got[i] >= spans[i].span || cqe->res == 0) {
                    if (spans[i].begin + got[i] < reqs[i].file_offset + reqs[i].bytes) {
                        io_uring_cq_advance(ring, seen);
                        throw std::runtime_error(std::string("moe: short io_uring read for ") + reqs[i].what);
                    }
                    done++;
                } else if (!submit_one(i)) {
                    io_uring_cq_advance(ring, seen);
                    throw std::runtime_error("moe: io_uring queue full while re-submitting a short read");
                }
            }
            io_uring_cq_advance(ring, seen);
        }
    } else
#endif
    {
        for (size_t i = 0; i < reqs.size(); ++i) {
            size_t g = 0;
            while (g < spans[i].span) {
                const ssize_t n = pread(fds_.at(reqs[i].shard), arena_ + spans[i].arena_off + g, spans[i].span - g, (off_t) (spans[i].begin + g));
                if (n < 0) {
                    throw std::runtime_error(std::string("moe: read error for ") + reqs[i].what);
                }
                if (n == 0) {
                    break;
                }
                g += (size_t) n;
                n_reads++;
            }
            if (spans[i].begin + g < reqs[i].file_offset + reqs[i].bytes) {
                throw std::runtime_error(std::string("moe: short read for ") + reqs[i].what);
            }
        }
    }

    uint64_t used = 0;
    for (size_t i = 0; i < reqs.size(); ++i) {
        memcpy(reqs[i].dst, arena_ + spans[i].arena_off + (reqs[i].file_offset - spans[i].begin), reqs[i].bytes);
        used += reqs[i].bytes;
    }
    reads_.fetch_add(n_reads, std::memory_order_relaxed);
    bytes_read_.fetch_add(total, std::memory_order_relaxed);
    bytes_used_.fetch_add(used, std::memory_order_relaxed);
#else
    (void) reqs;
    throw std::runtime_error("moe: IoUringExpertStore is implemented for Linux only");
#endif
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
            if (s.prefetched) {
                stats_.prefetch_wasted++;
                s.prefetched = false;
            }
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

int CachingExpertStore::resident_index(const ExpertDescriptor & e, bool is_prefetch) {
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
        slabs_[reserved].prefetched = is_prefetch;
        n_ready_++;
        stats_.bytes_inserted += e.total_bytes;
        if (is_prefetch) {
            stats_.prefetched++;
            stats_.bytes_prefetched += e.total_bytes;
        }
        i = reserved;
        cv_.notify_all();
    } else {
        if (is_prefetch) {
            // already resident: nothing to do, and not a prediction that helped
        } else {
            stats_.hits++;
            stats_.bytes_from_cache += e.total_bytes;
            if (slabs_[i].prefetched) {
                slabs_[i].prefetched = false;
                stats_.prefetch_useful++;
            }
        }
    }
    lru_touch_locked(i);
    slabs_[i].readers++;
    return i;
}

bool CachingExpertStore::contains(ExpertKey k) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = index_.find({ k.layer, k.expert });
    return it != index_.end() && slabs_[it->second].state == St::READY;
}

bool CachingExpertStore::prefetch(const ExpertDescriptor & e) {
    if (n_slots_ == 0 || e.total_bytes > bundle_bytes_) {
        return false;
    }
    if (contains(e.key)) {
        return false;
    }
    const int i = resident_index(e, /*is_prefetch=*/ true);
    std::lock_guard<std::mutex> lock(mtx_);
    slabs_[i].readers--;
    cv_.notify_all();
    return true;
}

const uint8_t * CachingExpertStore::acquire_bundle(const ExpertDescriptor & e) {
    if (n_slots_ == 0 || e.total_bytes > bundle_bytes_) {
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
    if (n_slots_ == 0 || e.total_bytes > bundle_bytes_) {
        // pass-through (cache disabled or a descriptor larger than a slab)
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

std::vector<ExpertKey> CachingExpertStore::resident_keys() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<ExpertKey> out;
    for (int i = head_; i >= 0; i = slabs_[i].next) {
        if (slabs_[i].state == St::READY) {
            out.push_back(ExpertKey{ slabs_[i].layer, slabs_[i].expert });
        }
    }
    return out;
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
        if (moe_slot_count(t) != (int64_t) n_slots) {
            throw std::runtime_error("moe: pool tensor slot axis != n_slots");
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
                fn_(i);
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

void FetchWorkers::start(size_t n, const std::function<void(size_t)> & fn) {
    std::lock_guard<std::mutex> lock(mtx_);
    fn_ = fn; n_ = n; next_ = 0; done_ = 0; errors_.clear(); gen_++;
    cv_.notify_all();
}

void FetchWorkers::wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    done_cv_.wait(lock, [&] { return done_ == n_; });
    fn_ = nullptr;
    if (!errors_.empty()) {
        std::exception_ptr e = errors_.front();
        errors_.clear();
        std::rethrow_exception(e);
    }
}

// Small fetch batches do not pay for the worker pool: start() takes a mutex and wakes every
// thread, and each task signals back. At decode the queue is ~3-15 bundles, so the dispatch can
// cost more than the reads save. 0 disables the inline path.
static size_t moe_inline_fetch_max() {
    static const size_t v = [] {
        const char * s = getenv("LLAMA_MOE_INLINE_FETCH");
        const int n = s ? atoi(s) : 0;
        return (size_t) (n > 0 ? n : 0);
    }();
    return v;
}

void FetchWorkers::run(size_t n, const std::function<void(size_t)> & fn) {
    start(n, fn);
    wait();
}

void LayerPool::set_device_backends(ggml_backend_t transfer, ggml_backend_t compute, ggml_backend_dev_t dev) {
    drain_transfers();
    if (event_) {
        ggml_backend_event_free(event_);
        event_ = nullptr;
    }
    transfer_ = transfer;
    compute_  = compute;
    if (transfer_ && compute_ && dev) {
        event_ = ggml_backend_event_new(dev);
        if (!event_) {
            transfer_ = nullptr;   // backend has no events: stay synchronous
        }
    }
}

void LayerPool::drain_transfers() {
    if (event_pending_ && event_) {
        ggml_backend_event_synchronize(event_);
        event_pending_ = false;
    }
    if (pending_cache_) {
        for (const auto & d : pending_release_) {
            pending_cache_->release_bundle(d);
        }
    }
    pending_release_.clear();
    pending_cache_ = nullptr;
    pending_staged_.clear();
}

LayerPool::~LayerPool() {
    drain_transfers();
    if (event_) {
        ggml_backend_event_free(event_);
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
    if (auto * cache = dynamic_cast<CachingExpertStore *>(&store)) {
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
        ggml_backend_tensor_set(tensors_[i], staging.data() + off, (size_t) slot * moe_slot_stride(tensors_[i]), d.slices[i].bytes);
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
        const uint64_t slot_bytes = moe_slot_stride(tensors_[i]);
        if (sl.bytes != slot_bytes) {
            throw std::runtime_error("moe: slice bytes (" + std::to_string(sl.bytes) + ") != pool slot bytes (" +
                                     std::to_string(slot_bytes) + ") for " + sl.name);
        }
    }
    fetch_into(store, d, slot, staging_);
    stats.bytes_read += d.total_bytes;
    s.state = SlotState::READY;
}

void LayerPool::load_many_uring(const ExpertIndex & idx, IoUringExpertStore & store,
                                const std::vector<std::pair<ExpertKey, int>> & work, PoolStats & stats) {
    std::vector<BatchRead> reqs;
    reqs.reserve(work.size() * tensors_.size());
    uint64_t bytes = 0;
    for (const auto & w : work) {
        const ExpertDescriptor & d = idx.get(w.first);
        if (d.slices.size() != tensors_.size()) {
            throw std::runtime_error("moe: descriptor slice count != pool tensor count");
        }
        Slot & sl = slots_[w.second];
        sl.state = SlotState::LOADING;
        sl.key   = w.first;
        sl.generation++;
        for (size_t k = 0; k < d.slices.size(); ++k) {
            const TensorSlice & s = d.slices[k];
            if (s.bytes != moe_slot_stride(tensors_[k])) {
                throw std::runtime_error("moe: slice bytes != pool slot bytes for " + s.name);
            }
            if (s.file_offset == 0) {
                throw std::runtime_error("moe: no file offset recorded for " + s.name);
            }
            // host pool: the slot's bytes live in host memory, read straight into them
            uint8_t * dst = (uint8_t *) tensors_[k]->data + (size_t) w.second * moe_slot_stride(tensors_[k]);
            reqs.push_back(BatchRead{ s.file_offset, s.bytes, dst, s.name.c_str(), s.shard });
        }
        bytes += d.total_bytes;
    }
    store.read_batch(reqs);   // no partial state on error: slots stay LOADING and the exception propagates
    for (const auto & w : work) {
        slots_[w.second].state = SlotState::READY;
    }
    stats.bytes_read += bytes;
}

void LayerPool::load_many(const ExpertIndex & idx, ExpertStore & store,
                          const std::vector<std::pair<ExpertKey, int>> & work, PoolStats & stats) {
    if (work.empty()) {
        return;
    }
    if (host_pool_) {
        // one thread drives the whole batch through the ring: no fetch workers, no per-thread arenas
        if (auto * uring = dynamic_cast<IoUringExpertStore *>(&store); uring && uring->available()) {
            load_many_uring(idx, *uring, work, stats);
            return;
        }
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
            if (d.slices[k].bytes != moe_slot_stride(tensors_[k])) {
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
    // Device pools with a transfer backend: issue each bundle's H2D copy as soon as it is resident, on the
    // transfer stream, while the other workers are still reading; then hand the compute stream an event.
    const bool async_dev = !host_pool_ && async_transfer();
    std::mutex done_mtx;
    std::condition_variable done_cv;
    std::vector<size_t> done_list;
    bool task_failed = false;
    auto task_notify = [&](size_t i) {
        try {
            task(i);
        } catch (...) {
            if (async_dev) {
                std::lock_guard<std::mutex> lock(done_mtx);
                task_failed = true;
                done_cv.notify_one();
            }
            throw;
        }
        if (async_dev) {
            std::lock_guard<std::mutex> lock(done_mtx);
            done_list.push_back(i);
            done_cv.notify_one();
        }
    };

    if (async_dev) {
        drain_transfers();                       // previous batch's copies are complete: release its slabs
        const bool inline_small = work.size() <= moe_inline_fetch_max();
        auto issue_copy = [&](size_t i) {
            const ExpertDescriptor & d = idx.get(work[i].first);
            const uint8_t * src = cache ? views[i] : staged[i].data();
            uint64_t off = 0;
            for (size_t k = 0; k < d.slices.size(); ++k) {
                ggml_backend_tensor_set_async(transfer_, tensors_[k], src + off, (size_t) work[i].second * tensors_[k]->nb[2], d.slices[k].bytes);
                off += d.slices[k].bytes;
            }
        };
        if (inline_small) {
            try {
                for (size_t i = 0; i < work.size(); ++i) {
                    task(i);
                    issue_copy(i);
                }
            } catch (...) {
                ggml_backend_synchronize(transfer_);
                if (cache) { for (size_t i = 0; i < work.size(); ++i) { if (views[i]) cache->release_bundle(idx.get(work[i].first)); } }
                throw;
            }
        } else {
        workers_->start(work.size(), task_notify);
        size_t issued = 0;
        try {
            while (issued < work.size()) {
                size_t i;
                {
                    std::unique_lock<std::mutex> lock(done_mtx);
                    done_cv.wait(lock, [&] { return task_failed || !done_list.empty(); });
                    if (done_list.empty()) {
                        break; // a worker failed: wait() below rethrows its error
                    }
                    i = done_list.back();
                    done_list.pop_back();
                }
                const ExpertDescriptor & d = idx.get(work[i].first);
                const uint8_t * src = cache ? views[i] : staged[i].data();
                uint64_t off = 0;
                for (size_t k = 0; k < d.slices.size(); ++k) {
                    ggml_backend_tensor_set_async(transfer_, tensors_[k], src + off, (size_t) work[i].second * moe_slot_stride(tensors_[k]), d.slices[k].bytes);
                    off += d.slices[k].bytes;
                }
                issued++;
            }
            workers_->wait();
        } catch (...) {
            workers_->wait();
            ggml_backend_synchronize(transfer_);
            if (cache) { for (size_t i = 0; i < work.size(); ++i) { if (views[i]) cache->release_bundle(idx.get(work[i].first)); } }
            throw;
        }
        }
        ggml_backend_event_record(event_, transfer_);
        ggml_backend_event_wait(compute_, event_);   // the graph's next nodes wait on the GPU, not the host
        event_pending_ = true;
        if (cache) {
            pending_cache_ = cache;
            for (const auto & w : work) { pending_release_.push_back(idx.get(w.first)); }
        } else {
            pending_staged_ = std::move(staged);
        }
        for (const auto & w : work) {
            slots_[w.second].state = SlotState::READY;
        }
        stats.bytes_read += bytes.load();
        return;
    }

    // no partial state is published on error: slots stay LOADING and the exception propagates
    if (work.size() <= moe_inline_fetch_max()) {
        for (size_t i = 0; i < work.size(); ++i) { task(i); }
    } else {
        workers_->run(work.size(), task);
    }

    if (!host_pool_) {
        for (size_t i = 0; i < work.size(); ++i) {
            const ExpertDescriptor & d = idx.get(work[i].first);
            if (cache) {
                uint64_t off = 0;
                for (size_t k = 0; k < d.slices.size(); ++k) {
                    ggml_backend_tensor_set(tensors_[k], views[i] + off, (size_t) work[i].second * moe_slot_stride(tensors_[k]), d.slices[k].bytes);
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
        if (k.layer != layer_) {
            throw std::runtime_error("moe: key for layer " + std::to_string(k.layer) + " sent to pool of layer " + std::to_string(layer_));
        }
        if (std::find(distinct.begin(), distinct.end(), k) == distinct.end()) {
            distinct.push_back(k);
        }
    }
    if (distinct.size() > slots_.size()) {
        const std::string who = "layer " + std::to_string(layer_);
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
