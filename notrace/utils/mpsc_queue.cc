#include "utils/mpsc_queue.h"
#include <algorithm>
#include <iostream>
#include "handlers/default_handler.h"
#include "handlers/kernel_launch.h"

namespace notrace {

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
  }

  size_t total_bytes = 0;

  for (ThreadLocalRingBuffer* buf : buffers_snapshot) {
    while (true) {
      BufferSpan span = buf->peek();

      if (span.size < sizeof(MPSCMessageHeader)) {
        break;
      }

      MPSCMessageHeader* header =
          reinterpret_cast<MPSCMessageHeader*>(span.data);

      size_t message_size = sizeof(MPSCMessageHeader) + header->size;

      if (span.size < message_size) {
        break;
      }

      if (header->api_type < this->consumers_.size()) {
        MessageConsumer consumer = this->consumers_[header->api_type];
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
      }

      buf->advance(message_size);
      total_bytes += message_size;
    }
  }
  return total_bytes;
}

void MPSCMessageQueue::registerConsumers() {
  static kernel_launch::KernelLaunchConsumer kernelLaunchConsumer;
  this->registerConsumer(nvbit_api_cuda_t::API_CUDA_cuLaunchKernel,
                         &kernelLaunchConsumer);
}

}  // namespace notrace
