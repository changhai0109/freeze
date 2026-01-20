#ifndef __NOTRACE_KERNEL_LAUNCH_H__
#define __NOTRACE_KERNEL_LAUNCH_H__

#include <stdint.h>
#include <cstdio>
#include <vector>

#include "cuda.h"
#include "cuda_event_handlers.h"
#include "nvbit.h"

namespace notrace {
namespace kernel_launch {

const uint8_t MESSAGE_TYPE_KERNEL_START = 1;
const uint8_t MESSAGE_TYPE_KERNEL_END = 2;

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  uint64_t launchId;
  uint64_t gpuStartCycles;

  CUstream stream;
  CUcontext context;

  uint32_t gridX, gridY, gridZ;
  uint32_t blockX, blockY, blockZ;
  uint32_t sharedMemBytes;
} KernelLaunchStartInfo;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  uint64_t launchId;
  uint64_t gpuEndCycles;
} KernelLaunchEndInfo;
#pragma pack(pop)

}  // namespace kernel_launch
}  // namespace notrace

#endif  // __NOTRACE_KERNEL_LAUNCH_H__
