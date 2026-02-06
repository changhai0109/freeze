#include "handlers/stream_create.h"
#include <cstdint>
#include "handlers/launch_kernel.h"
#include "utils/mpsc_queue.h"
#include "utils/stream_event_mapper.h"

namespace notrace {
namespace stream_create {

StreamEventMapper& streamEventMapper = StreamEventMapper::getInstance();
static thread_local MessageWritter messageWritter;

void StreamCreateProducer::onStartHook(CUcontext ctx, const char* name,
                                       void* params, CUresult* pStatus) {
  // No action needed on stream creation start
}

void StreamCreateProducer::onEndHook(CUcontext ctx, const char* name,
                                     void* params, CUresult* pStatus) {
  if constexpr (notrace::kernel_launch::HOTSPOT_MODE) {
    return;  // Skip processing to reduce overhead in hotspot mode
  }

  StreamCreateMsg* msg = messageWritter.reserve<StreamCreateMsg>(
      nvbit_api_cuda_t::API_CUDA_cuStreamCreate);
  cuStreamCreate_params* p = (cuStreamCreate_params*)params;
  msg->stream = *(p->phStream);
  msg->flags = static_cast<uint32_t>(p->Flags);
  msg->priority = INT32_MIN;
  msg->tid = std::this_thread::get_id();
  messageWritter.commit();
}

void StreamCreateWithPriorityProducer::onStartHook(CUcontext ctx,
                                                   const char* name,
                                                   void* params,
                                                   CUresult* pStatus) {
  // No action needed on stream creation start
}

void StreamCreateWithPriorityProducer::onEndHook(CUcontext ctx,
                                                 const char* name, void* params,
                                                 CUresult* pStatus) {
  if constexpr (notrace::kernel_launch::HOTSPOT_MODE) {
    return;  // Skip processing to reduce overhead in hotspot mode
  }
  StreamCreateMsg* msg = messageWritter.reserve<StreamCreateMsg>(
      nvbit_api_cuda_t::API_CUDA_cuStreamCreate);
  cuStreamCreateWithPriority_params* p =
      (cuStreamCreateWithPriority_params*)params;
  msg->stream = *(p->phStream);
  msg->flags = 0;
  msg->priority = p->priority;
  msg->tid = std::this_thread::get_id();
  messageWritter.commit();
}

void StreamCreateConsumer::processImpl(void* data, size_t size) {
  StreamCreateMsg* msg = reinterpret_cast<StreamCreateMsg*>(data);
  // Record the stream start event
  streamEventMapper.recordStreamStart(msg->stream, msg->tid);
  printf(
      "[StreamCreateConsumer] Stream created: stream=%p, flags=%u, priority=%d "
      "in thread %lu\n",
      msg->stream, msg->flags, msg->priority,
      std::hash<std::thread::id>{}(msg->tid));
}

void streamCreateProducerWrapper(CUcontext ctx, int is_exit, const char* name,
                                 void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  static StreamCreateProducer producer;
  producer.apiHook(ctx, is_exit, name, params, pStatus);
}

void streamCreateWithPriorityProducerWrapper(CUcontext ctx, int is_exit,
                                             const char* name, void* params,
                                             CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  static StreamCreateWithPriorityProducer producer;
  producer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace stream_create
}  // namespace notrace
