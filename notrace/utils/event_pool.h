#ifndef NOTRACE_UTILS_EVENT_POOL_H__
#define NOTRACE_UTILS_EVENT_POOL_H__

#include <cuda_runtime.h>
#include <vector>

// ----------------------------------
// CUDA event helpers
// ----------------------------------
typedef struct {
  cudaEvent_t start;
  cudaEvent_t end;
} CudaEventPair;

class CudaEventPool {
 public:
  CudaEventPair acquire();
  void release(CudaEventPair pair);
  ~CudaEventPool();

  static CudaEventPair createEventPair();

 private:
  std::vector<CudaEventPair> pool_;
};

#endif  // NOTRACE_UTILS_EVENT_POOL_H__
