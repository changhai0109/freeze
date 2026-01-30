#ifndef __NOTRACE_UTILS_MPSC_QUEUE_H__
#define __NOTRACE_UTILS_MPSC_QUEUE_H__

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "common.h"
#include "handlers/base_handler.h"
#include "handlers/default_handler.h"
#include "utils/ring_buffer.h"

namespace notrace {

extern std::atomic<uint64_t> global_mpsc_buffer_id;
#pragma pack(push, 1)
struct MPSCMessageHeaderNoId {
  nvbit_api_cuda_t api_type;
  uint32_t size;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct MPSCMessageHeaderWithId {
  nvbit_api_cuda_t api_type;
  uint64_t id;
  uint32_t size;
};
#pragma pack(pop)

using MPSCMessageHeader =
    std::conditional_t<notrace::debug::ENABLE_DEBUG_LOGS,
                       MPSCMessageHeaderWithId, MPSCMessageHeaderNoId>;

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
      if constexpr (notrace::debug::ENABLE_MPSC_DEBUG_LOGS)
        printf(
            "[MPSC] Initialized ThreadLocalRingBuffer id=%lu for thread %lu\n",
            buffer_->id(), std::this_thread::get_id());
    }

    size_t totalSize = sizeof(T) + sizeof(MPSCMessageHeader);

    void* reserved = buffer_->reserve(totalSize);
    if (reserved == nullptr) {
      if constexpr (notrace::debug::ENABLE_MPSC_DEBUG_LOGS)
        printf("[MPSC] Failed to reserve %zu bytes in buffer id=%lu\n",
               totalSize, buffer_->id());
      return nullptr;
    }
    MPSCMessageHeader* header = reinterpret_cast<MPSCMessageHeader*>(reserved);
    header->api_type = type;
    if constexpr (notrace::debug::ENABLE_MPSC_DEBUG_LOGS) {
      auto* h = reinterpret_cast<MPSCMessageHeaderWithId*>(header);
      h->id = global_mpsc_buffer_id.fetch_add(1, std::memory_order_relaxed);
    }
    header->size = sizeof(T);
    if constexpr (notrace::debug::ENABLE_MPSC_DEBUG_LOGS) {
      auto* h = reinterpret_cast<MPSCMessageHeaderWithId*>(header);
      printf("[MPSC] Reserved %zu bytes in buffer id=%lu, message id=%lu\n",
             totalSize, buffer_->id(), h->id);
    }
    return reinterpret_cast<T*>(header + 1);
  }

  void commit() {
    if (buffer_ == nullptr) {
      if constexpr (notrace::debug::ENABLE_MPSC_DEBUG_LOGS)
        printf("[MPSC] Error: commit called before initialize\n");
      return;
    }
    buffer_->commit();
  }

  uint64_t getBufferId();

 private:
  void initialize() {
    buffer_ = MPSCMessageQueue::getInstance().getThreadLocalBuffer();
  };

  ThreadLocalRingBuffer* buffer_ = nullptr;
};

}  // namespace notrace

#endif
