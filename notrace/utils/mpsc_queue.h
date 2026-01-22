#ifndef __NOTRACE_UTILS_MPSC_QUEUE_H__
#define __NOTRACE_UTILS_MPSC_QUEUE_H__

#include <condition_variable>
#include <mutex>
#include <vector>
#include "common.h"
#include "handlers/base_handler.h"
#include "handlers/default_handler.h"
#include "utils/ring_buffer.h"

namespace notrace {

#pragma pack(push, 1)
struct MPSCMessageHeader {
  nvbit_api_cuda_t api_type;
  uint32_t size;
};
#pragma pack(pop)

class MPSCMessageQueue {
 public:
  static MPSCMessageQueue& getInstance() {
    static MPSCMessageQueue instance;
    return instance;
  }

  ThreadLocalRingBuffer* getThreadLocalBuffer();

  using MessageConsumer = TraceConsumer*;
  void registerConsumer(nvbit_api_cuda_t type, MessageConsumer consumer);
  void registerConsumers();
  size_t processUpdates();

 private:
  MPSCMessageQueue();
  ~MPSCMessageQueue();
  MPSCMessageQueue(const MPSCMessageQueue&) = delete;
  MPSCMessageQueue& operator=(const MPSCMessageQueue&) = delete;

  MessageConsumer defaultConsumer_;

  std::vector<ThreadLocalRingBuffer*> buffers_;
  std::mutex registryMutex_;
  std::vector<MessageConsumer> consumers_;
};

class MessageWritter {
 public:
  template <typename T>
  T* reserve(nvbit_api_cuda_t type) {
    if (buffer_ == nullptr) [[unlikely]] {
      initialize();
    }

    size_t totalSize = sizeof(T) + sizeof(MPSCMessageHeader);

    void* reserved = buffer_->reserve(totalSize);
    if (reserved == nullptr) {
      return nullptr;
    }
    MPSCMessageHeader* header = reinterpret_cast<MPSCMessageHeader*>(reserved);
    header->api_type = type;
    header->size = sizeof(T);
    return reinterpret_cast<T*>(header + 1);
  }

  void commit() { buffer_->commit(); }

 private:
  void initialize() {
    buffer_ = MPSCMessageQueue::getInstance().getThreadLocalBuffer();
  };

  ThreadLocalRingBuffer* buffer_ = nullptr;
};

}  // namespace notrace

#endif
