#include "handlers/kernel_launch.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

#include "cuda.h"
#include "generated_cuda_meta.h"
#include "nvbit.h"
#include "utils/cuda_safecall.h"
#include "utils/event_pool.h"
#include "utils/mpsc_queue.h"
#include "utils/string_store.h"

namespace notrace {
namespace kernel_launch {

// Global singletons/statics
std::atomic<uint64_t> launchIdCounter{0};
thread_local uint64_t currentLaunchId = 0;

CudaEventPool& cudaEventPool = CudaEventPool::getInstance();
StringStore& stringStore = StringStore::getInstance();

// Thread-local writer for the PRODUCER threads (Application threads)
static thread_local MessageWritter messageWritter;

static KernelLaunchProducer globalKernelProducer;

// Helpers for ID generation
inline uint64_t getAndSaveLaunchId() {
  currentLaunchId = launchIdCounter.fetch_add(1, std::memory_order_relaxed);
  return currentLaunchId;
}

inline uint64_t getCurrentLaunchId() {
  return currentLaunchId;
}

// =============================================================================
// KernelLaunchProducer Implementation
// =============================================================================

void KernelLaunchProducer::onStartHook(CUcontext ctx, const char* name,
                                       void* params, CUresult* pStatus) {
  uint64_t launchId = getAndSaveLaunchId();

  // Reserve space in the queue
  auto* msg = messageWritter.reserve<KernelLaunchStartInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);

  if (msg == nullptr) [[unlikely]] {
    return;
  }

  cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

  const char* kernelName = "unknown_kernel";
  cuFuncGetName(&kernelName, p->f);

  msg->messageType = MESSAGE_TYPE_KERNEL_START;
  msg->kernelNameId = stringStore.getStringId(std::string(kernelName));
  msg->launchId = launchId;
  // Acquire CUDA event for timing
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

  // Record start event
  CUDA_SAFECALL(
      cudaEventRecordWithFlags(msg->startEvent, msg->stream, cudaEventDefault));

  messageWritter.commit();
}

void KernelLaunchProducer::onEndHook(CUcontext ctx, const char* name,
                                     void* params, CUresult* pStatus) {
  uint64_t launchId = getCurrentLaunchId();

  auto* msg = messageWritter.reserve<KernelLaunchEndInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);

  if (msg == nullptr) [[unlikely]] {
    return;
  }

  cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

  msg->messageType = MESSAGE_TYPE_KERNEL_END;
  msg->launchId = launchId;
  msg->endEvent = cudaEventPool.acquire();

  // Record end event
  CUDA_SAFECALL(
      cudaEventRecordWithFlags(msg->endEvent, p->hStream, cudaEventDefault));

  messageWritter.commit();
}

// =============================================================================
// KernelLaunchConsumer Implementation
// =============================================================================

KernelLaunchConsumer::~KernelLaunchConsumer() {
  // Cleanup any pending start info structs to avoid memory leaks
  for (auto& pair : pending_launches_) {
    // Technically we should also release events here if they weren't processed
    // cudaEventPool.release(pair.second->startEvent);
    delete pair.second;
  }
  pending_launches_.clear();
}

void KernelLaunchConsumer::process(void* data, size_t size) {
  uint8_t messageType = *(uint8_t*)data;

  switch (messageType) {
    case MESSAGE_TYPE_KERNEL_START:
      if (size != sizeof(KernelLaunchStartInfo)) {
        assert(false && "Invalid size for KernelLaunchStartInfo");
        return;
      }
      processStart(reinterpret_cast<KernelLaunchStartInfo*>(data));
      break;
    case MESSAGE_TYPE_KERNEL_END:
      if (size != sizeof(KernelLaunchEndInfo)) {
        assert(false && "Invalid size for KernelLaunchEndInfo");
        return;
      }
      processEnd(reinterpret_cast<KernelLaunchEndInfo*>(data));
      break;
    case MESSAGE_TYPE_KERNEL_PROCESSED:
      if (size != sizeof(KernelLaunchRecord)) {
        assert(false && "Invalid size for KernelLaunchRecord");
        return;
      }
      processRecord(reinterpret_cast<KernelLaunchRecord*>(data));
      break;
    default:
      assert(false && "Unknown message type in KernelLaunchConsumer");
      break;
  }
}

void KernelLaunchConsumer::processStart(KernelLaunchStartInfo* startInfo) {
  if (pending_launches_.find(startInfo->launchId) != pending_launches_.end()) {
    assert(false && "Duplicate launch ID detected");
    return;
  }

  // Persist the start info.
  // We MUST copy it because the original 'msg' pointer points to the RingBuffer,
  // which will be overwritten as soon as we return from process().
  KernelLaunchStartInfo* startInfoCopy = new KernelLaunchStartInfo(*startInfo);
  pending_launches_[startInfo->launchId] = startInfoCopy;
}

void KernelLaunchConsumer::processEnd(KernelLaunchEndInfo* endInfo) {
  if (cudaEventQuery(endInfo->endEvent) == cudaErrorNotReady) {
    // The end event has not yet been recorded by the GPU.
    // Re-enqueue the end message for later processing.
    void* newMsg = messageWritter.reserve<KernelLaunchEndInfo>(
        nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);
    assert(newMsg && "Failed to re-reserve end message");
    *static_cast<KernelLaunchEndInfo*>(newMsg) = *endInfo;
    messageWritter.commit();  // Re-commit the same message
    // std::this_thread::sleep_for(
    //     std::chrono::microseconds(100));  // Backoff to avoid busy looping
    // printf("Re-enqueuing end message for launchId %lu\n", endInfo->launchId);
    return;
  }

  auto it = pending_launches_.find(endInfo->launchId);
  if (it == pending_launches_.end()) {
    // This happens if the ring buffer overwrote the start message before we read it,
    // or if tracing started mid-execution.
    // Release the end event to avoid leaks.
    cudaEventPool.release(endInfo->endEvent);
    return;
  }

  KernelLaunchStartInfo* startInfo = it->second;

  // 1. Calculate GPU Duration
  // Note: cudaEventElapsedTime syncs the CPU thread until GPU records the event.
  // Use caution if you want strictly non-blocking consumers.
  float durationMs = 0.0f;
  CUDA_SAFECALL(cudaEventElapsedTime(&durationMs, startInfo->startEvent,
                                     endInfo->endEvent));

  // 2. Generate the "Processed" Record
  // The Consumer acts as a Producer here! It writes back to the queue.
  // Note: We use the same traceWriter mechanism.
  auto* record = messageWritter.reserve<KernelLaunchRecord>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);

  if (record) {
    record->messageType = MESSAGE_TYPE_KERNEL_PROCESSED;
    record->kernelNameId = startInfo->kernelNameId;
    record->launchId = startInfo->launchId;
    record->gpuStartCycles = 0;  // Or fetch clock()
    record->gpuEndCycles = static_cast<uint64_t>(durationMs * 1e6);  // ns

    record->stream = startInfo->stream;
    record->context = startInfo->context;
    record->gridX = startInfo->gridX;
    record->gridY = startInfo->gridY;
    record->gridZ = startInfo->gridZ;
    record->blockX = startInfo->blockX;
    record->blockY = startInfo->blockY;
    record->blockZ = startInfo->blockZ;
    record->sharedMemBytes = startInfo->sharedMemBytes;

    messageWritter.commit();
  }

  // 3. Cleanup
  cudaEventPool.release(startInfo->startEvent);
  cudaEventPool.release(endInfo->endEvent);
  delete startInfo;
  pending_launches_.erase(it);
}

void KernelLaunchConsumer::processRecord(KernelLaunchRecord* msg) {
  printf(
      "KernelLaunchRecord: Name=%s, Duration=%.3f ns\n, grid=(%u,%u,%u), "
      "block=(%u,%u,%u)\n",
      stringStore.getStringFromId(msg->kernelNameId).c_str(),
      static_cast<float>(msg->gpuEndCycles), msg->gridX, msg->gridY, msg->gridZ,
      msg->blockX, msg->blockY, msg->blockZ);
}

void kernelLaunchHookWrapper(CUcontext ctx, int is_exit, const char* name,
                             void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  globalKernelProducer.kernelLaunchHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace kernel_launch
}  // namespace notrace
