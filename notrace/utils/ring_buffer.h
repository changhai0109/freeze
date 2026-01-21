#ifndef __NOTRACE_UTILS_RING_BUFFER_H__
#define __NOTRACE_UTILS_RING_BUFFER_H__

#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "common.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

namespace notrace {

struct BufferSpan {
  uint8_t* data;
  size_t size;
};

class ThreadLocalRingBuffer {
 public:
  static constexpr size_t BUFFER_SIZE = 10 * 1024 * 1024;  // 10MB

  uint8_t* data;

  alignas(64) std::atomic<size_t> commitPos;
  alignas(64) std::atomic<size_t> consumerPos;

  size_t reservedPos;
  bool pendingCommit;

  ThreadLocalRingBuffer();

  ~ThreadLocalRingBuffer();

  void initMagicBuffer();

  void* reserve(size_t size);

  template <typename T>
  T* reserve() {
    void* ptr = this->reserve(sizeof(T));
    if (ptr == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<T*>(ptr);
  }

  void commit();

  BufferSpan peek();

  void advance(size_t bytes_processed);
};

}  // namespace notrace

#endif  // __NOTRACE_UTILS_RING_BUFFER_H__
