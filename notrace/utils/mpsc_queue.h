#ifndef __NOTRACE_UTILS_MPSC_QUEUE_H__
#define __NOTRACE_UTILS_MPSC_QUEUE_H__

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "nvbit.h"

namespace notrace {

class MPSCMessageBuffer {
 public:
  static constexpr size_t BUFFER_SIZE = 10 * 1024 * 1024;

  uint8_t* data;

  std::atomic<size_t> commitPos;
  std::atomic<size_t> consumerPos;

  size_t reservedPos;
  size_t cachedConsumerPos;
  bool pendingCommit;

  MPSCMessageBuffer() {
    initMagicBuffer();

    commitPos.store(0, std::memory_order_relaxed);
    consumerPos.store(0, std::memory_order_relaxed);
    reservedPos = 0;
    cachedConsumerPos = 0;
    pendingCommit = false;
  }

  ~MPSCMessageBuffer() { munmap(data, BUFFER_SIZE * 2); }

  void initMagicBuffer() {
    int fd = memfd_create("nvbit_ring_buffer", 0);
    if (fd < 0)
      throw std::runtime_error("memfd_create failed");

    if (ftruncate(fd, BUFFER_SIZE) < 0)
      throw std::runtime_error("ftruncate failed");

    uint8_t* base_ptr = (uint8_t*)mmap(nullptr, 2 * BUFFER_SIZE, PROT_NONE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ptr == MAP_FAILED)
      throw std::runtime_error("Reserve mmap failed");

    void* ptr1 = mmap(base_ptr, BUFFER_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_FIXED, fd, 0);

    void* ptr2 = mmap(base_ptr + BUFFER_SIZE, BUFFER_SIZE,
                      PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);

    if (ptr1 == MAP_FAILED || ptr2 == MAP_FAILED)
      throw std::runtime_error("Magic mmap failed");

    close(fd);
    data = base_ptr;
  }

  void* reserve(size_t size) {
    if (pendingCommit)
      return nullptr;
    if (size > BUFFER_SIZE / 2)
      return nullptr;  // Sanity check

    size_t head = reservedPos;
    size_t tail = cachedConsumerPos;

    size_t used = head - tail;

    if (used + size > BUFFER_SIZE) {
      tail = consumerPos.load(std::memory_order_acquire);
      cachedConsumerPos = tail;
      used = head - tail;

      if (used + size > BUFFER_SIZE) {
        return nullptr;  // Buffer is legitimately full
      }
    }

    reservedPos += size;
    pendingCommit = true;

    return data + (head % BUFFER_SIZE);
  }

  void commit() {
    if (!pendingCommit) [[unlikely]]
      throw std::runtime_error("No commit pending");
    commitPos.store(reservedPos, std::memory_order_release);
    pendingCommit = false;
  }
};

}  // namespace notrace

#endif  // __NOTRACE_UTILS_MPSC_QUEUE_H__
