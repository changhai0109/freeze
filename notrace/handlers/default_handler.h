#ifndef NOTRACE_HANDLERS_DEFAULT_HANDLER_H__
#define NOTRACE_HANDLERS_DEFAULT_HANDLER_H__

#include "handlers/base_handler.h"
#include "utils/string_store.h"

namespace notrace {
namespace default_handler {

const uint8_t MESSAGE_TYPE_START = 1;
const uint8_t MESSAGE_TYPE_END = 2;

#pragma pack(push, 1)
struct DefaultHandlerParams {
  uint8_t messageType;
  StringId apiNameId;
};
#pragma pack(pop)

class DefaultHandlerProducer : public TraceProducer {
 public:
  DefaultHandlerProducer() = default;
  ~DefaultHandlerProducer() override = default;

 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;
  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class DefaultHandlerConsumer : public TraceConsumer {
 public:
  DefaultHandlerConsumer() = default;
  ~DefaultHandlerConsumer() override = default;

 private:
  void processImpl(void* data, size_t size) override;
};

void defaultHandlerHookWrapper(CUcontext ctx, int is_exit, const char* name,
                               void* params, CUresult* pStatus);

void setCBID(nvbit_api_cuda_t cbid);

}  // namespace default_handler
}  // namespace notrace

#endif  // NOTRACE_HANDLERS_DEFAULT_HANDLER_H__
