#ifndef __NOTRACE_KERNEL_LAUNCH_H__
#define __NOTRACE_KERNEL_LAUNCH_H__

#include <cuda_runtime.h>
#include <tuple>
#include "cuda_event_handlers.h"
#include "generated_cuda_meta.h"
#include "nvbit.h"

namespace notrace {

typedef struct {
  const char* kernel_name;
  uint64_t launch_id;

  CUstream stream;
  CUcontext context;

  uint64_t gpu_start_cycles;
  uint64_t gpu_end_cycles;

  uint32_t grid_x, grid_y, grid_z;
  uint32_t block_x, block_y, block_z;
  uint32_t shared_mem_bytes;
} kernel_launch_info_t;

typedef struct {
  uint64_t launch_id;
  size_t record_index;
} pending_kernel_t;

typedef struct {
  cudaEvent_t start;
  cudaEvent_t end;
} cuda_event_pair_t;

typedef struct {
  cudaEvent_t t0;
  uint64_t host_t0_ns;
} cuda_stream_clock_t;

typedef struct {
  cuda_event_pair_t events;
  kernel_launch_info_t* rec;
} KernelDonePayload;

class CudaEventPool {
 public:
  cuda_event_pair_t acquire();
  void release(cuda_event_pair_t pair);
  ~CudaEventPool();

 private:
  std::vector<cuda_event_pair_t> pool_;
  static cuda_event_pair_t create_event_pair();
};

void handle_kernel_launch(CUcontext ctx, int device_id, const char* kernel_name,
                          void* kernel_params, CUresult* pResult);

};  // namespace notrace

#endif  // __NOTRACE_KERNEL_LAUNCH_H__
