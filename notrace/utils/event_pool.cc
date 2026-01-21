#include "utils/event_pool.h"
#include <iostream>
#include "utils/cuda_safecall.h"

cudaEvent_t CudaEventPool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);  // Lock happens here

  if (!pool_.empty()) {
    cudaEvent_t event = pool_.back();
    pool_.pop_back();
    return event;
  }

  // Pool empty? Create new batch.
  for (int i = 0; i < INITIAL_POOL_SIZE - 1; ++i) {
    cudaEvent_t event;
    CUDA_SAFECALL(cudaEventCreate(&event));  // Add error checking in real code
    pool_.push_back(event);
  }
  cudaEvent_t event = pool_.back();
  pool_.pop_back();
  return event;
}

void CudaEventPool::release(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex_);  // Lock happens here
  pool_.push_back(event);
}

CudaEventPool::~CudaEventPool() {
  for (auto e : pool_) {
    cudaEventDestroy(e);
  }
}
