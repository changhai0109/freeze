#include "handlers/ctx_get_current.h"
#include "handlers/launch_kernel.h"
#include "utils/mpsc_queue.h"
#include "utils/stream_event_mapper.h"

namespace notrace {
namespace ctx_get_current {

thread_local MessageWritter messageWritter;
StreamEventMapper& streamEventMapper = StreamEventMapper::getInstance();

void CtxGetCurrentProducer::onStartHook(CUcontext ctx, const char* name,
                                        void* params, CUresult* pStatus) {
  // No action needed on start
}

void CtxGetCurrentProducer::onEndHook(CUcontext ctx, const char* name,
                                      void* params, CUresult* pStatus) {
  CtxGetCurrentMsg* msg = messageWritter.reserve<CtxGetCurrentMsg>(
      nvbit_api_cuda_t::API_CUDA_cuCtxGetCurrent);
  msg->tid = std::this_thread::get_id();
  messageWritter.commit();
}

void CtxGetCurrentConsumer::processImpl(void* data, size_t size) {
  CtxGetCurrentMsg* msg = reinterpret_cast<CtxGetCurrentMsg*>(data);
  printf("[CtxGetCurrentConsumer] cuCtxGetCurrent called in thread %lu\n",
         std::hash<std::thread::id>{}(msg->tid));
  streamEventMapper.recordStreamStart(NULL, msg->tid);
}

void ctxGetCurrentProducerWrapper(CUcontext ctx, int is_exit, const char* name,
                                  void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  static CtxGetCurrentProducer producer;
  producer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace ctx_get_current
}  // namespace notrace
