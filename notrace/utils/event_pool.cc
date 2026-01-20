#include "utils/event_pool.h"
#include "utils/cuda_safecall.h"

CudaEventPair CudaEventPool::acquire() {
  // Caller must synchronize externally if used across threads.
  if (pool_.empty())
    return createEventPair();

  CudaEventPair pair = pool_.back();
  pool_.pop_back();
  return pair;
}

void CudaEventPool::release(CudaEventPair pair) {
  pool_.push_back(pair);
}

CudaEventPool::~CudaEventPool() {
  for (auto& pair : pool_) {
    CUDA_SAFECALL(cudaEventDestroy(pair.start));
    CUDA_SAFECALL(cudaEventDestroy(pair.end));
  }
  pool_.clear();
}

CudaEventPair CudaEventPool::createEventPair() {
  CudaEventPair pair{};

  CUDA_SAFECALL(cudaEventCreateWithFlags(&pair.start, cudaEventDefault));
  CUDA_SAFECALL(cudaEventCreateWithFlags(&pair.end, cudaEventDefault));

  return pair;
}
