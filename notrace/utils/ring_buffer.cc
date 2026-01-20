#include "utils/ring_buffer.h"

namespace notrace {

ThreadLocalRingBuffer::ThreadLocalRingBuffer() {
  this->initMagicBuffer();
  this->commitPos.store(0, std::memory_order_relaxed);
  this->consumerPos.store(0, std::memory_order_relaxed);
  this->reservedPos = 0;
  this->pendingCommit = false;
}

ThreadLocalRingBuffer::~ThreadLocalRingBuffer() {
  munmap(data, BUFFER_SIZE * 2);
}

void ThreadLocalRingBuffer::initMagicBuffer() {
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
  void* ptr2 = mmap(base_ptr + BUFFER_SIZE, BUFFER_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, fd, 0);

  if (ptr1 == MAP_FAILED || ptr2 == MAP_FAILED)
    throw std::runtime_error("Magic mmap failed");

  close(fd);
  data = base_ptr;
}

void* ThreadLocalRingBuffer::reserve(size_t size) {
  if (pendingCommit)
    return nullptr;
  if (size > BUFFER_SIZE / 2)
    return nullptr;

  size_t head = reservedPos;
  size_t tail = consumerPos.load(std::memory_order_acquire);

  if ((head - tail) + size > BUFFER_SIZE) {
    return nullptr;  // Buffer Full
  }

  reservedPos += size;
  pendingCommit = true;

  return data + (head % BUFFER_SIZE);
}

void ThreadLocalRingBuffer::commit() {
  if (!pendingCommit) [[unlikely]]
    throw std::runtime_error(
        "No pending commit to commit, check if reversed before commit or "
        "nested commit!");
  commitPos.store(reservedPos, std::memory_order_release);
  pendingCommit = false;
}

BufferSpan ThreadLocalRingBuffer::peek() {
  size_t comm = commitPos.load(std::memory_order_acquire);
  size_t cons = consumerPos.load(std::memory_order_relaxed);

  if (comm == cons) {
    return {nullptr, 0};
  }

  size_t available = comm - cons;

  uint8_t* ptr = data + (cons % BUFFER_SIZE);

  return {ptr, available};
}

void ThreadLocalRingBuffer::advance(size_t bytes_processed) {
  if (bytes_processed == 0)
    return;

  size_t current_cons = consumerPos.load(std::memory_order_relaxed);
  consumerPos.store(current_cons + bytes_processed, std::memory_order_release);
}

}  // namespace notrace
