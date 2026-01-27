#include "utils/mpsc_queue.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include "common.h"
#include "handlers/default_handler.h"
#include "handlers/launch_kernel.h"
#include "handlers/mem_alloc.h"
#include "handlers/mem_free.h"
#include "nvbit.h"

namespace notrace {

std::atomic<uint64_t> global_mpsc_buffer_id{0};

MPSCMessageQueue::~MPSCMessageQueue() {
  std::lock_guard<std::mutex> lock(registryMutex_);
  for (auto buf : buffers_) {
    delete buf;
  }
  buffers_.clear();

  if constexpr (notrace::default_handler::ENABLE_DEFAULT_HANDLER)
    delete this->defaultConsumer_;
}

MPSCMessageQueue::MPSCMessageQueue() {
  if constexpr (notrace::default_handler::ENABLE_DEFAULT_HANDLER) {
    defaultConsumer_ = new notrace::default_handler::DefaultHandlerConsumer();
  } else {
    defaultConsumer_ = nullptr;
  }
}

ThreadLocalRingBuffer* MPSCMessageQueue::getThreadLocalBuffer() {
  ThreadLocalRingBuffer* newBuffer = new ThreadLocalRingBuffer();

  {
    std::lock_guard<std::mutex> lock(registryMutex_);
    buffers_.push_back(newBuffer);
    if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
      printf("[MPSC] new ThreadLocalRingBuffer id=%lu created, in thread %lu\n",
             newBuffer->id(),
             std::hash<std::thread::id>{}(std::this_thread::get_id()));
  }

  return newBuffer;
}

void MPSCMessageQueue::registerConsumer(nvbit_api_cuda_t type,
                                        MessageConsumer consumer) {
  std::lock_guard<std::mutex> lock(registryMutex_);

  if (type >= consumers_.size()) {
    size_t required = type + 1;
    size_t current = consumers_.size();
    size_t new_size = std::max({required, current * 2, size_t(64)});
    consumers_.resize(new_size, nullptr);
  }

  consumers_[type] = consumer;
}

size_t MPSCMessageQueue::processUpdates() {
  std::vector<ThreadLocalRingBuffer*> buffers_snapshot;
  {
    std::lock_guard<std::mutex> lock(this->registryMutex_);
    buffers_snapshot = this->buffers_;

    if constexpr (notrace::debug::ENABLE_DEBUG_LOGS) {
      printf("[MPSC] Processing %zu buffers, threadId=%lu\n",
             buffers_snapshot.size(),
             std::hash<std::thread::id>{}(std::this_thread::get_id()));
      for (auto buf : buffers_snapshot) {
        printf("[MPSC]   Buffer id=%lu\n", buf->id());
      }
    }
  }

  size_t total_bytes = 0;

  for (ThreadLocalRingBuffer* buf : buffers_snapshot) {
    while (true) {
      BufferSpan span = buf->peek();

      if (span.data == nullptr || span.size == 0) {
        if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
          printf("[MPSC] No more data to process in buffer id=%lu\n",
                 buf->id());
        break;
      }
      if (span.size < sizeof(MPSCMessageHeader)) {
        assert(span.size < sizeof(MPSCMessageHeader) &&
               "Not enough data for MPSCMessageHeader");
        break;
      }

      MPSCMessageHeader* header =
          reinterpret_cast<MPSCMessageHeader*>(span.data);

      size_t message_size = sizeof(MPSCMessageHeader) + header->size;

      if (span.size < message_size) {
        assert(span.size < message_size &&
               "Not enough data for complete message");
        break;
      }
      if constexpr (notrace::debug::ENABLE_DEBUG_LOGS) {
        auto* h = reinterpret_cast<MPSCMessageHeaderWithId*>(header);
        printf(
            "[MPSC] Processing message of type %d, size %u, message id=%lu "
            "from "
            "buffer id=%lu\n",
            header->api_type, header->size, h->id, buf->id());
      }

      MessageConsumer consumer = nullptr;
      if (header->api_type < this->consumers_.size()) {
        consumer = this->consumers_[header->api_type];
      }

      // printf("Processing message of type %d, size %u\n, consumer %p",
      //  header->api_type, header->size, (void*)consumer);

      // fflush(stdout);
      if (consumer == nullptr) {
        if constexpr (notrace::default_handler::ENABLE_DEFAULT_HANDLER)
          consumer = this->defaultConsumer_;
      }
      if (consumer != nullptr) {
        void* payload = reinterpret_cast<void*>(header + 1);
        consumer->process(payload, header->size);
      }

      buf->advance(message_size);
      total_bytes += message_size;
    }
  }
  return total_bytes;
}

uint64_t MessageWritter::getBufferId() {
  if (buffer_ == nullptr) {
    return UINT64_MAX;
  }
  return buffer_->id();
}

void MPSCMessageQueue::registerConsumers() {
  static kernel_launch::LaunchKernelConsumer kernelLaunchConsumer;
  static mem_alloc::MemAllocConsumer memAllocConsumer;
  static mem_free::MemFreeConsumer memFreeConsumer;
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuLaunchKernel,
                         &kernelLaunchConsumer);
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuMemAlloc_v2,
                         &memAllocConsumer);
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuMemAlloc,
                         &memAllocConsumer);
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuMemFree_v2,
                         &memFreeConsumer);
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuMemFree,
                         &memFreeConsumer);
}

}  // namespace notrace
