#include "handlers/default_handler.h"
#include "nvbit.h"
#include "utils/mpsc_queue.h"
#include "utils/string_store.h"

namespace notrace {
namespace default_handler {

static thread_local MessageWritter messageWritter;
static thread_local DefaultHandlerProducer defaultHandlerProducer;
static thread_local nvbit_api_cuda_t cbid;
StringStore& stringStore = StringStore::getInstance();

void DefaultHandlerProducer::onStartHook(CUcontext ctx, const char* name,
                                         void* params, CUresult* pStatus) {
  // printf("DefaultHandlerProducer::onStartHook called for %s\n", name);
  // Reserve space in the queue
  auto* msg = messageWritter.reserve<DefaultHandlerParams>(cbid);

  if (msg == nullptr) {
    return;
  }

  msg->messageType = MESSAGE_TYPE_START;
  msg->apiNameId = stringStore.getStringId(name);

  messageWritter.commit();
}

void DefaultHandlerProducer::onEndHook(CUcontext ctx, const char* name,
                                       void* params, CUresult* pStatus) {
  // printf("DefaultHandlerProducer::onEndHook called for %s\n", name);
  // Reserve space in the queue
  auto* msg = messageWritter.reserve<DefaultHandlerParams>(cbid);

  if (msg == nullptr) {
    return;
  }

  msg->messageType = MESSAGE_TYPE_END;
  msg->apiNameId = stringStore.getStringId(name);

  messageWritter.commit();
}

void DefaultHandlerConsumer::processImpl(void* data, size_t size) {
  // printf("size: %zu, expected: %zu\n", size, sizeof(DefaultHandlerParams));
  assert(size == sizeof(DefaultHandlerParams) &&
         "Invalid size for DefaultHandlerParams");
  DefaultHandlerParams* msg = reinterpret_cast<DefaultHandlerParams*>(data);
  const char* apiName = "unknown_api";
  apiName = stringStore.getStringFromId(msg->apiNameId).c_str();

  switch (msg->messageType) {
    case MESSAGE_TYPE_START:
      // printf("CUDA API ENTER: %s", apiName);
      break;
    case MESSAGE_TYPE_END:
      // printf("CUDA API EXIT: %s", apiName);
      break;
    default:
      assert(false && "Invalid message type in DefaultHandlerConsumer");
      break;
  }
}

void defaultHandlerHookWrapper(CUcontext ctx, int is_exit, const char* name,
                               void* params, CUresult* pStatus) {
  // Forward the C-style call to the C++ Class instance
  defaultHandlerProducer.kernelLaunchHook(ctx, is_exit, name, params, pStatus);
}

void setCBID(nvbit_api_cuda_t cbid_) {
  cbid = cbid_;
}

}  // namespace default_handler
}  // namespace notrace
