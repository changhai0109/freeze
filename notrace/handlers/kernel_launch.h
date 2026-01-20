#ifndef __NOTRACE_KERNEL_LAUNCH_H__
#define __NOTRACE_KERNEL_LAUNCH_H__

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <vector>

#include "cuda_event_handlers.h"
#include "generated_cuda_meta.h"
#include "nvbit.h"

namespace notrace {

typedef struct {
  const char* kernel_name;
  uint64_t launch_id;

  CUstream stream;
  CUcontext context;

  uint64_t gpuStartCycles;
  uint64_t gpuEndCycles;

  uint32_t gridX, gridY, gridZ;
  uint32_t blockX, blockY, blockZ;
  uint32_t sharedMemBytes;
} KernelLaunchInfo;

// Per-stream anchor used to map CUDA event time -> host time
typedef struct {
  cudaEvent_t t0;
  uint64_t host_t0_ns;
} CudaStreamClock;

void handleKernelLaunch(CUcontext ctx, int device_id, const char* kernel_name,
                        void* kernel_params, CUresult* pResult);

}  // namespace notrace

#endif  // __NOTRACE_KERNEL_LAUNCH_H__
