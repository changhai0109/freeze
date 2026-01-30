#include "handlers/mem_alloc.h"
#include <cassert>
#include "tracker/memory_tracker.h"
#include "utils/mpsc_queue.h"

namespace notrace {
namespace mem_alloc {

static thread_local MessageWritter messageWritter;
static MemAllocProducer globalMemAllocProducer;
static memory_tracker::MemoryTracker& memoryTracker =
    memory_tracker::MemoryTracker::getInstance();

void MemAllocProducer::onStartHook(CUcontext ctx, const char* name,
                                   void* params, CUresult* pStatus) {
  // do nothing
}
void MemAllocProducer::onEndHook(CUcontext ctx, const char* name, void* params,
                                 CUresult* pStatus) {
  if (pStatus && *pStatus != CUDA_SUCCESS) {
    // Memory allocation failed, do not log
    // const char* errStr;
    // cuGetErrorName(*pStatus, &errStr);
    // printf("cuMemAlloc failed with error: %s\n", errStr);
    return;
  }

  auto* msg = messageWritter.reserve<MemAllocRecord>(
      nvbit_api_cuda_t::API_CUDA_cuMemAlloc_v2);
  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve MemAllocRecord");
    return;
  }

  cuMemAlloc_v2_params* p = (cuMemAlloc_v2_params*)params;
  msg->ptr = *(p->dptr);
  // printf("ptr=%p, *ptr=%p\n", p->dptr, (void*)(*p->dptr));
  msg->size = p->bytesize;
  messageWritter.commit();
}

void MemAllocConsumer::processImpl(void* data, size_t size) {
  assert(size == sizeof(MemAllocRecord) && "Invalid size for MemAllocRecord");
  auto* msg = reinterpret_cast<MemAllocRecord*>(data);
  if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
    printf("MemAllocv2Record: ptr=%p, size=%zu\n", (void*)(msg->ptr),
           msg->size);
  memoryTracker.recordAllocation((void*)(msg->ptr), msg->size,
                                 memory_tracker::Location::DEVICE);
}

void memAllocHookWrapper(CUcontext ctx, int is_exit, const char* name,
                         void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  globalMemAllocProducer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace mem_alloc
}  // namespace notrace
