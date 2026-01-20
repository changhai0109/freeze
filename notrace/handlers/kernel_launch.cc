// handlers/kernel_launch.cc
#include "handlers/kernel_launch.h"
#include <atomic>
#include <cassert>
#include <memory>
#include "generated_cuda_meta.h"
#include "nvbit.h"
#include "utils/event_pool.h"
#include "utils/mpsc_queue.h"
#include "utils/string_store.h"

namespace notrace {
namespace kernel_launch {

std::atomic<uint64_t> launchIdCounter{0};
thread_local uint64_t currentLaunchId = 0;
MPSCMessageQueue& messageQueue = MPSCMessageQueue::getInstance();
CudaEventPool& cudaEventPool = CudaEventPool::getInstance();
StringStore& stringStore = StringStore::getInstance();
static thread_local TraceProducer traceProducer;

static thread_local std::unordered_map<uint64_t, KernelLaunchStartInfo*>
    launchIdToStartInfoMap;

inline uint64_t getAndSaveLaunchId() {
  currentLaunchId = launchIdCounter.fetch_add(1, std::memory_order_relaxed);
  return currentLaunchId;
}

inline uint64_t getCurrentLaunchId() {
  return currentLaunchId;
}

void hookStart(CUcontext ctx, const char* kernel_name, void* kernel_params,
               CUresult* pResult) {
  uint64_t launchId = getAndSaveLaunchId();
  KernelLaunchStartInfo* msg = traceProducer.reserve<KernelLaunchStartInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);
  if (msg == nullptr) [[unlikely]] {
    // Failed to reserve message, skip logging
    // TODO: should log an error?
    return;
  }

  cuLaunchKernel_params* p = (cuLaunchKernel_params*)kernel_params;

  msg->messageType = MESSAGE_TYPE_KERNEL_START;
  msg->kernelNameId = stringStore.getStringId(std::string(kernel_name));
  msg->launchId = launchId;
  msg->startEvent = cudaEventPool.acquire();

  msg->stream = p->hStream;
  msg->context = ctx;

  msg->gridX = p->gridDimX;
  msg->gridY = p->gridDimY;
  msg->gridZ = p->gridDimZ;
  msg->blockX = p->blockDimX;
  msg->blockY = p->blockDimY;
  msg->blockZ = p->blockDimZ;
  msg->sharedMemBytes = p->sharedMemBytes;

  cudaEventRecordWithFlags(msg->startEvent, msg->stream, cudaEventDefault);
  traceProducer.commit();
}

void hookEnd(CUcontext ctx, const char* kernel_name, void* kernel_params,
             CUresult* pResult) {
  uint64_t launchId = getCurrentLaunchId();
  KernelLaunchEndInfo* msg = traceProducer.reserve<KernelLaunchEndInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);
  if (msg == nullptr) [[unlikely]] {
    // Failed to reserve message, skip logging
    // TODO: should log an error?
    return;
  }

  cuLaunchKernel_params* p = (cuLaunchKernel_params*)kernel_params;

  msg->messageType = MESSAGE_TYPE_KERNEL_END;
  msg->launchId = launchId;
  msg->endEvent = cudaEventPool.acquire();

  cudaEventRecordWithFlags(msg->endEvent, p->hStream, cudaEventDefault);
  traceProducer.commit();
}

void processStart(void* data) {
  KernelLaunchStartInfo* msg = reinterpret_cast<KernelLaunchStartInfo*>(data);

  if (msg == nullptr) [[unlikely]] {
    // shouldn;t happen
    assert(false);
    return;
  }
  if (launchIdToStartInfoMap.find(msg->launchId) !=
      launchIdToStartInfoMap.end()) {
    // duplicate launchId, shouldn't happen
    assert(false);
    return;
  } else {
    // save the start info for later end processing
    KernelLaunchStartInfo* startInfoCopy = new KernelLaunchStartInfo(*msg);
    launchIdToStartInfoMap[msg->launchId] = startInfoCopy;
  }
}

void processEnd(void* data) {
  KernelLaunchEndInfo* endInfo = reinterpret_cast<KernelLaunchEndInfo*>(data);

  if (endInfo == nullptr) [[unlikely]] {
    // shouldn;t happen
    assert(false);
    return;
  }

  auto it = launchIdToStartInfoMap.find(endInfo->launchId);
  if (it == launchIdToStartInfoMap.end()) {
    // missing start info, shouldn't happen
    assert(false);
    return;
  }

  KernelLaunchStartInfo* startInfo = it->second;
  // get GPU timing info
  float durationMs = 0.0f;
  cudaEventElapsedTime(&durationMs, startInfo->startEvent, endInfo->endEvent);

  // create record
  KernelLaunchRecord* record = traceProducer.reserve<KernelLaunchRecord>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);
  record->messageType = MESSAGE_TYPE_KERNEL_PROCESSED;
  record->kernelName = stringStore.getStringFromId(startInfo->kernelNameId);
  record->launchId = startInfo->launchId;
  record->gpuStartCycles = 0;  // For device cycle driven timing, not used here
  record->gpuEndCycles = durationMs * 1e6;  // convert ms to ns;
  record->stream = startInfo->stream;
  record->context = startInfo->context;
  record->gridX = startInfo->gridX;
  record->gridY = startInfo->gridY;
  record->gridZ = startInfo->gridZ;
  record->blockX = startInfo->blockX;
  record->blockY = startInfo->blockY;
  record->blockZ = startInfo->blockZ;
  record->sharedMemBytes = startInfo->sharedMemBytes;
  traceProducer.commit();

  // cleanup
  cudaEventPool.release(startInfo->startEvent);
  cudaEventPool.release(endInfo->endEvent);
  delete startInfo;
  launchIdToStartInfoMap.erase(it);
}

void kernelLaunchHook(CUcontext ctx, int is_exit, const char* name,
                      void* params, CUresult* pStatus) {
  if (is_exit) {
    hookEnd(ctx, name, params, pStatus);
  } else {
    hookStart(ctx, name, params, pStatus);
  }
}

void process(void* data) {
  uint8_t messageType = *(uint8_t*)data;
  switch (messageType) {
    case MESSAGE_TYPE_KERNEL_START:
      processStart(data);
      break;
    case MESSAGE_TYPE_KERNEL_END:
      processEnd(data);
      break;
    case MESSAGE_TYPE_KERNEL_PROCESSED:
      // processed message, nothing to do
      break;
    default:
      // unknown message type
      assert(false);
      break;
  }
}

};  // namespace kernel_launch
};  // namespace notrace
