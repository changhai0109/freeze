#include "utils/mpsc_queue.h"
#include <algorithm>
#include <iostream>

namespace notrace {

MPSCMessageQueue::~MPSCMessageQueue() {
  std::lock_guard<std::mutex> lock(registryMutex_);
  for (auto buf : buffers_) {
    delete buf;
  }
  buffers_.clear();
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
        if (consumer != nullptr) {
          void* payload = reinterpret_cast<void*>(header + 1);
          consumer(payload, header->size);
        }
      }

      buf->advance(message_size);
      total_bytes += message_size;
    }
  }
  return total_bytes;
}

inline void TraceProducer::initialize() {
  buffer_ = MPSCMessageQueue::getInstance().getThreadLocalBuffer();
}
}  // namespace notrace
