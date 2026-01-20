#ifndef __NOTRACE_KERNEL_LAUNCH_H__
#define __NOTRACE_KERNEL_LAUNCH_H__

#include <stdint.h>
#include <cstdio>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_event_handlers.h"
#include "nvbit.h"
#include "utils/string_store.h"

namespace notrace {
namespace kernel_launch {

const uint8_t MESSAGE_TYPE_KERNEL_START = 1;
const uint8_t MESSAGE_TYPE_KERNEL_END = 2;
const uint8_t MESSAGE_TYPE_KERNEL_PROCESSED = 4;

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  StringId kernelNameId;
  uint64_t launchId;
  cudaEvent_t startEvent;

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
  cudaEvent_t endEvent;
} KernelLaunchEndInfo;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  std::string kernelName;
  uint64_t launchId;
  uint64_t gpuStartCycles;
  uint64_t gpuEndCycles;

  CUstream stream;
  CUcontext context;

  uint32_t gridX, gridY, gridZ;
  uint32_t blockX, blockY, blockZ;
  uint32_t sharedMemBytes;
} KernelLaunchRecord;
#pragma pack(pop)

void kernelLaunchHook(CUcontext ctx, int is_exit, const char* name,
                      void* params, CUresult* pStatus);

void process(void* data);

}  // namespace kernel_launch
}  // namespace notrace

#endif  // __NOTRACE_KERNEL_LAUNCH_H__
