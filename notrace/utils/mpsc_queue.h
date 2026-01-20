#ifndef __NOTRACE_UTILS_MPSC_QUEUE_H__
#define __NOTRACE_UTILS_MPSC_QUEUE_H__

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "nvbit.h"
#include "utils/ring_buffer.h"

namespace notrace {

using MessageConsumer = void (*)(void* data, size_t size);

struct alignas(8) MPSCMessageHeader {
  nvbit_api_cuda_t api_type;
  uint32_t size;
};

class MPSCMessageQueue {
 private:
  std::vector<ThreadLocalRingBuffer*> buffers_;
  std::mutex registryMutex_;

  std::vector<MessageConsumer> consumers_;

 public:
  MPSCMessageQueue() = default;

  ~MPSCMessageQueue();

  // producer API
  ThreadLocalRingBuffer* getThreadLocalBuffer();

  template <typename T = void>
  T* reserveMessage(nvbit_api_cuda_t type) {
    return reinterpret_cast<T*>(this->reserveBytes(type, sizeof(T)));
  }

  void* reserveBytes(nvbit_api_cuda_t type, size_t payloadSize);

  void commitMessage();

  // consumer API
  void registerConsumer(nvbit_api_cuda_t type, MessageConsumer consumer);

  size_t processUpdates();
};

}  // namespace notrace

#endif  // __NOTRACE_UTILS_MPSC_QUEUE_H__
