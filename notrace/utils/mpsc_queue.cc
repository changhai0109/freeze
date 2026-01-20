#include "utils/mpsc_queue.h"
#include "utils/ring_buffer.h"

namespace notrace {
MPSCMessageQueue::~MPSCMessageQueue() {
  std::lock_guard<std::mutex> lock(registryMutex_);
  for (auto buf : buffers_) {
    delete buf;
  }
}

ThreadLocalRingBuffer* MPSCMessageQueue::getThreadLocalBuffer() {
  static thread_local ThreadLocalRingBuffer* myBuffer = nullptr;

  if (myBuffer == nullptr) {
    myBuffer = new ThreadLocalRingBuffer();

    std::lock_guard<std::mutex> lock(registryMutex_);
    buffers_.push_back(myBuffer);
  }

  return myBuffer;
}

void* MPSCMessageQueue::reserveBytes(nvbit_api_cuda_t type,
                                     size_t payloadSize) {
  ThreadLocalRingBuffer* buf = this->getThreadLocalBuffer();

  size_t total_size = sizeof(MPSCMessageHeader) + payloadSize;
  void* ptr = buf->reserve(total_size);
  if (ptr == nullptr) {
    return nullptr;
  }

  MPSCMessageHeader* header = reinterpret_cast<MPSCMessageHeader*>(ptr);
  header->api_type = type;
  header->size = static_cast<uint32_t>(payloadSize);

  return reinterpret_cast<void*>(header + 1);
}

void MPSCMessageQueue::commitMessage() {
  ThreadLocalRingBuffer* buf = this->getThreadLocalBuffer();
  buf->commit();
}

void MPSCMessageQueue::registerConsumer(nvbit_api_cuda_t type,
                                        MessageConsumer consumer) {
  std::lock_guard<std::mutex> lock(registryMutex_);
  if (type >= consumers_.size()) {
    size_t required = type + 1;
    size_t doubled = consumers_.size() * 2;
    constexpr size_t min_start = 64;
    size_t new_size = std::max({required, doubled, min_start});
    consumers_.resize(new_size, nullptr);
  }
}

size_t MPSCMessageQueue::processUpdates() {
  std::unique_lock<std::mutex> lock(this->registryMutex_);
  size_t total_bytes = 0;
  for (ThreadLocalRingBuffer* buf : this->buffers_) {
    while (true) {
      BufferSpan span = buf->peek();
      if (span.size < sizeof(MPSCMessageHeader)) {
        assert(false);  // the message should always be complete with a header
        break;
      }

      MPSCMessageHeader* header =
          reinterpret_cast<MPSCMessageHeader*>(span.data);
      size_t message_size = sizeof(MPSCMessageHeader) + header->size;

      if (span.size < message_size) {
        assert(false);
        break;  // the message should always shorter than the available size
      }

      MessageConsumer consumer = nullptr;
      if (this->consumers_.size() > header->api_type) {
        consumer = this->consumers_[header->api_type];
      }
      if (consumer != nullptr) {
        void* payload = reinterpret_cast<void*>(header + 1);
        consumer(payload, header->size);
      }

      buf->advance(message_size);
      total_bytes += message_size;
    }
  }
  return total_bytes;
}

}  // namespace notrace
