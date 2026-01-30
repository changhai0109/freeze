#include "handlers/mem_free.h"
#include "common.h"
#include "generated_cuda_meta.h"
#include "tracker/memory_tracker.h"
#include "utils/mpsc_queue.h"

namespace notrace {
namespace mem_free {
static thread_local MessageWritter messageWritter;
static MemFreeProducer globalMemFreeProducer;

void MemFreeProducer::onStartHook(CUcontext ctx, const char* name, void* params,
                                  CUresult* pStatus) {
  // do nothing
  cuMemFree_v2_params* p = (cuMemFree_v2_params*)params;
  if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
    printf("MemFree called for ptr=%p\n", (void*)(p->dptr));
}

void MemFreeProducer::onEndHook(CUcontext ctx, const char* name, void* params,
                                CUresult* pStatus) {
  if (pStatus && *pStatus != CUDA_SUCCESS) {
    // Memory free failed, do not log
    return;
  }
  cuMemFree_v2_params* p = (cuMemFree_v2_params*)params;

  if (!p->dptr) {
    // Freeing a null pointer, nothing to log
    return;
  }

  auto* msg = messageWritter.reserve<MemFreeRecord>(
      nvbit_api_cuda_t::API_CUDA_cuMemFree_v2);
  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve MemFreeRecord");
    return;
  }

  msg->ptr = p->dptr;
  messageWritter.commit();
}

void MemFreeConsumer::processImpl(void* data, size_t size) {
  assert(size == sizeof(MemFreeRecord) && "Invalid size for MemFreeRecord");
  auto* msg = reinterpret_cast<MemFreeRecord*>(data);
  // if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
  printf("MemFreev2Record: ptr=%p\n", (void*)(msg->ptr));
  memory_tracker::MemoryTracker::getInstance().recordDeallocation(
      (void*)(msg->ptr));
}

void memFreeHookWrapper(CUcontext ctx, int is_exit, const char* name,
                        void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  globalMemFreeProducer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace mem_free
}  // namespace notrace
