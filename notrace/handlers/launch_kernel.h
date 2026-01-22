#ifndef __NOTRACE_HANDLERS_KERNEL_LAUNCH_H__
#define __NOTRACE_HANDLERS_KERNEL_LAUNCH_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "handlers/base_handler.h"  // Include the new base definitions
#include "nvbit.h"
#include "utils/string_store.h"

namespace notrace {
namespace kernel_launch {

// Message Type Constants
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
} LaunchKernelStartInfo;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  uint64_t launchId;
  cudaEvent_t endEvent;
} LaunchKernelEndInfo;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct {
  uint8_t messageType;
  StringId kernelNameId;
  uint64_t launchId;
  uint64_t gpuStartCycles;
  uint64_t gpuEndCycles;

  CUstream stream;
  CUcontext context;

  uint32_t gridX, gridY, gridZ;
  uint32_t blockX, blockY, blockZ;
  uint32_t sharedMemBytes;
} LaunchKernelRecord;
#pragma pack(pop)

/**
 * Producer: Hooks into CUDA driver calls, captures arguments, 
 * acquires events, and writes Start/End messages to the queue.
 */
class LaunchKernelProducer : public TraceProducer {
 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;

  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

/**
 * Consumer: Reads Start/End messages.
 * It waits for the End message, calculates GPU duration using events,
 * and produces a final "Processed" record.
 */
class LaunchKernelConsumer : public TraceConsumer {
 public:
  LaunchKernelConsumer() = default;
  ~LaunchKernelConsumer();  // Cleanup leftover map entries

  void processImpl(void* data, size_t size) override;

 private:
  void processStart(LaunchKernelStartInfo* msg);
  void processEnd(LaunchKernelEndInfo* msg);
  void processRecord(LaunchKernelRecord* msg);

  // Correlation map to match Start events with End events.
  // Owned by the consumer instance (single-threaded access assumed in processUpdates)
  std::unordered_map<uint64_t, LaunchKernelStartInfo*> pending_launches_;
};

void launchKernelHookWrapper(CUcontext ctx, int is_exit, const char* name,
                             void* params, CUresult* pStatus);

}  // namespace kernel_launch
}  // namespace notrace

#endif  // __NOTRACE_HANDLERS_KERNEL_LAUNCH_H__
