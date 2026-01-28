#include "handlers/launch_kernel.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <thread>

#include "cuda.h"
#include "generated_cuda_meta.h"
#include "nvbit.h"
#include "tracker/memory_tracker.h"
#include "utils/cuda_safecall.h"
#include "utils/event_pool.h"
#include "utils/mpsc_queue.h"
#include "utils/string_store.h"

namespace notrace {
namespace kernel_launch {

// Global singletons/statics
std::atomic<uint64_t> launchIdCounter{0};
thread_local uint64_t currentLaunchId = 0;
thread_local std::unordered_map<CUfunction, CachedKernelParamData>
    kernelParamMetadataCache;

CudaEventPool& cudaEventPool = CudaEventPool::getInstance();
StringStore& stringStore = StringStore::getInstance();

// Thread-local writer for the PRODUCER threads (Application threads)
static thread_local MessageWritter messageWritter;

static LaunchKernelProducer globalKernelProducer;

static memory_tracker::MemoryTracker& memoryTracker =
    memory_tracker::MemoryTracker::getInstance();

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

void LaunchKernelProducer::onStartHook(CUcontext ctx, const char* name,
                                       void* params, CUresult* pStatus) {
  // printf("KernelLaunchProducer::onStartHook called for %s\n", name);
  uint64_t launchId = getAndSaveLaunchId();

  // Reserve space in the queue
  auto* msg = messageWritter.reserve<LaunchKernelStartInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);

  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve message in LaunchKernelProducer");
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

  CachedKernelParamData const* cachedParams;
  auto it = kernelParamMetadataCache.find(p->f);
  if (it != kernelParamMetadataCache.end()) {
    cachedParams = &it->second;
  } else {
    // Query parameter metadata from CUDA
    CachedKernelParamData newCahceData;
    newCahceData.numParams = 0;

    for (uint8_t i = 0; i < MAX_KERNEL_ARGS; i++) {
      size_t offset = 0;
      size_t size = 0;
      CUresult res = cuFuncGetParamInfo(p->f, i, &offset, &size);
      if (res != CUDA_SUCCESS) {
        break;
      }
      newCahceData.params[i].bufferOffset = offset;
      newCahceData.params[i].size = size;
      newCahceData.numParams++;
    }
    cachedParams =
        &kernelParamMetadataCache.emplace(p->f, newCahceData).first->second;
  }
  msg->numArgs = 0;
  msg->currentArgDataOffset = 0;
  if (p->kernelParams != nullptr && cachedParams->numParams > 0) {
    for (uint8_t i = 0; i < cachedParams->numParams; i++) {
      size_t paramSize = cachedParams->params[i].size;

      if (msg->currentArgDataOffset + paramSize > MAX_TOTAL_ARG_DATA_BYTES)
          [[unlikely]] {
        // Exceeded max buffer size
        assert(false && "Exceeded max total arg data bytes");
        break;
      }

      // Copy parameter data from the kernelParams array
      void* sourceParamPtr = reinterpret_cast<uint8_t**>(p->kernelParams)[i];
      void* destParamPtr = &(msg->argDataBuffer[msg->currentArgDataOffset]);
      memcpy(destParamPtr, sourceParamPtr, paramSize);

      // Fill in the KernelArgInfo
      msg->argInfos[msg->numArgs].bufferOffset = msg->currentArgDataOffset;
      msg->argInfos[msg->numArgs].size = static_cast<uint16_t>(paramSize);
      msg->numArgs++;
      msg->currentArgDataOffset += static_cast<uint16_t>(paramSize);
    }
  }

  // Record start event
  CUDA_SAFECALL(
      cudaEventRecordWithFlags(msg->startEvent, msg->stream, cudaEventDefault));

  messageWritter.commit();
}

void LaunchKernelProducer::onEndHook(CUcontext ctx, const char* name,
                                     void* params, CUresult* pStatus) {
  // printf("KernelLaunchProducer::onEndHook called for %s\n", name);
  uint64_t launchId = getCurrentLaunchId();

  auto* msg = messageWritter.reserve<LaunchKernelEndInfo>(
      nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);

  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve message in LaunchKernelProducer");
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

LaunchKernelConsumer::~LaunchKernelConsumer() {
  // Cleanup any pending start info structs to avoid memory leaks
  for (auto& pair : pending_launches_) {
    // Technically we should also release events here if they weren't processed
    // cudaEventPool.release(pair.second->startEvent);
    delete pair.second;
  }
  pending_launches_.clear();
}

void LaunchKernelConsumer::processImpl(void* data, size_t size) {
  uint8_t messageType = *(uint8_t*)data;

  switch (messageType) {
    case MESSAGE_TYPE_KERNEL_START:
      if (size != sizeof(LaunchKernelStartInfo)) {
        assert(false && "Invalid size for KernelLaunchStartInfo");
        return;
      }
      processStart(reinterpret_cast<LaunchKernelStartInfo*>(data));
      break;
    case MESSAGE_TYPE_KERNEL_END:
      if (size != sizeof(LaunchKernelEndInfo)) {
        assert(false && "Invalid size for KernelLaunchEndInfo");
        return;
      }
      processEnd(reinterpret_cast<LaunchKernelEndInfo*>(data));
      break;
    case MESSAGE_TYPE_KERNEL_PROCESSED:
      if (size != sizeof(LaunchKernelRecord)) {
        assert(false && "Invalid size for KernelLaunchRecord");
        return;
      }
      processRecord(reinterpret_cast<LaunchKernelRecord*>(data));
      break;
    default:
      assert(false && "Unknown message type in KernelLaunchConsumer");
      break;
  }
}

void LaunchKernelConsumer::processStart(LaunchKernelStartInfo* startInfo) {
  if (pending_launches_.find(startInfo->launchId) != pending_launches_.end()) {
    assert(false && "Duplicate launch ID detected");
    return;
  }

  // Persist the start info.
  // We MUST copy it because the original 'msg' pointer points to the RingBuffer,
  // which will be overwritten as soon as we return from process().
  LaunchKernelStartInfo* startInfoCopy = new LaunchKernelStartInfo(*startInfo);
  pending_launches_[startInfo->launchId] = startInfoCopy;
}

void LaunchKernelConsumer::processEnd(LaunchKernelEndInfo* endInfo) {
  if (cudaEventQuery(endInfo->endEvent) == cudaErrorNotReady) {
    // The end event has not yet been recorded by the GPU.
    // Re-enqueue the end message for later processing.
    void* newMsg = messageWritter.reserve<LaunchKernelEndInfo>(
        nvbit_api_cuda_t::API_CUDA_cuLaunchKernel);
    assert(newMsg && "Failed to re-reserve end message");
    *static_cast<LaunchKernelEndInfo*>(newMsg) = *endInfo;
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

  LaunchKernelStartInfo* startInfo = it->second;

  // 1. Calculate GPU Duration
  // Note: cudaEventElapsedTime syncs the CPU thread until GPU records the event.
  // Use caution if you want strictly non-blocking consumers.
  float durationMs = 0.0f;
  CUDA_SAFECALL(cudaEventElapsedTime(&durationMs, startInfo->startEvent,
                                     endInfo->endEvent));

  // 1.5 Optional: Analyze Kernel Arguments
  analyzeKernelArgs(startInfo);

  // 2. Generate the "Processed" Record
  // The Consumer acts as a Producer here! It writes back to the queue.
  // Note: We use the same traceWriter mechanism.
  auto* record = messageWritter.reserve<LaunchKernelRecord>(
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

void LaunchKernelConsumer::processRecord(LaunchKernelRecord* msg) {
  printf(
      "KernelLaunchRecord: Name=%s, Duration=%.3f ns, grid=(%u,%u,%u), "
      "block=(%u,%u,%u)\n",
      stringStore.getStringFromId(msg->kernelNameId).c_str(),
      static_cast<float>(msg->gpuEndCycles), msg->gridX, msg->gridY, msg->gridZ,
      msg->blockX, msg->blockY, msg->blockZ);
}

void launchKernelHookWrapper(CUcontext ctx, int is_exit, const char* name,
                             void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  globalKernelProducer.apiHook(ctx, is_exit, name, params, pStatus);
}

void LaunchKernelConsumer::analyzeKernelArgs(LaunchKernelStartInfo* msg) {
  if (msg->numArgs == 0) {
    printf("No kernel arguments captured.\n");
    return;
  }

  printf("Kernel Arguments (total %u):\n", msg->numArgs);
  for (uint8_t i = 0; i < msg->numArgs; i++) {
    KernelArgInfo& argInfo = msg->argInfos[i];
    size_t size = argInfo.size;
    if (size == 8) {
      // maybe a pointer
      if (memoryTracker.exists(
              *(void**)(msg->argDataBuffer + argInfo.bufferOffset))) {
        void* ptr = *(void**)(msg->argDataBuffer + argInfo.bufferOffset);
        size_t allocSize = memoryTracker.getAllocationSize(ptr);
        printf("  Arg %u: ptr=%p (allocated size=%zu bytes)\n", i, ptr,
               allocSize);
      } else {
        uint64_t val = *(uint64_t*)(msg->argDataBuffer + argInfo.bufferOffset);
        printf("  Arg %u: uint64_t=%lu\n", i, val);
      }
    } else if (size == 4) {
      uint32_t val = *(uint32_t*)(msg->argDataBuffer + argInfo.bufferOffset);
      printf("  Arg %u: uint32_t=%u\n", i, val);
    } else if (size == 2) {
      uint16_t val = *(uint16_t*)(msg->argDataBuffer + argInfo.bufferOffset);
      printf("  Arg %u: uint16_t=%u\n", i, val);
    } else if (size == 1) {
      uint8_t val = *(uint8_t*)(msg->argDataBuffer + argInfo.bufferOffset);
      printf("  Arg %u: uint8_t=%u\n", i, val);
    } else {
      printf("  Arg %u: [unhandled size=%zu bytes]=", i, size);
      for (size_t b = 0; b < size; b++) {
        // Just dump raw bytes
        uint8_t byteVal =
            *(uint8_t*)(msg->argDataBuffer + argInfo.bufferOffset + b);
        printf("%02x ", byteVal);
      }
      printf("\n");
    }
  }
}

}  // namespace kernel_launch
}  // namespace notrace
