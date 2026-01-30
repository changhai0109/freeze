#ifndef NOTRACE_HANDLERS_CTX_GET_CURRENT_H
#define NOTRACE_HANDLERS_CTX_GET_CURRENT_H

#include <thread>
#include "handlers/base_handler.h"

namespace notrace {
namespace ctx_get_current {

struct CtxGetCurrentMsg {
  std::thread::id tid;
};

class CtxGetCurrentProducer : public TraceProducer {
 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;

  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class CtxGetCurrentConsumer : public TraceConsumer {
 public:
  void processImpl(void* data, size_t size) override;
};

void ctxGetCurrentProducerWrapper(CUcontext ctx, int is_exit, const char* name,
                                  void* params, CUresult* pStatus);

}  // namespace ctx_get_current
}  // namespace notrace

#endif  // NOTRACE_HANDLERS_CTX_GET_CURRENT_H
