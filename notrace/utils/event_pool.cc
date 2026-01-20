#include "utils/event_pool.h"
#include <iostream>

cudaEvent_t CudaEventPool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);  // Lock happens here

  if (!pool_.empty()) {
    cudaEvent_t evt = pool_.back();
    pool_.pop_back();
    return evt;
  }

  // Pool empty? Create new one.
  cudaEvent_t new_evt;
  cudaEventCreate(&new_evt);  // Add error checking in real code
  return new_evt;
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
