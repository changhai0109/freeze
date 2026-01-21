#ifndef NOTRACE_UTILS_EVENT_POOL_H__
#define NOTRACE_UTILS_EVENT_POOL_H__

#include <cuda_runtime.h>
#include <mutex>
#include <vector>

class CudaEventPool {
 public:
  constexpr static int INITIAL_POOL_SIZE = 32;
  CudaEventPool(const CudaEventPool&) = delete;
  CudaEventPool& operator=(const CudaEventPool&) = delete;

  static CudaEventPool& getInstance() {
    static CudaEventPool instance;
    return instance;
  }

  cudaEvent_t acquire();
  void release(cudaEvent_t event);

 private:
  CudaEventPool() = default;
  ~CudaEventPool();

  std::vector<cudaEvent_t> pool_;
  std::mutex mutex_;
};

#endif
