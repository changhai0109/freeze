#ifndef __NOTRACE_HANDLERS_BASE_HANDLER_H__
#define __NOTRACE_HANDLERS_BASE_HANDLER_H__

#include <cstdint>
#include "cuda.h"
#include "utils/api_call_flaggers.h"

namespace notrace {
class TraceConsumer {
 public:
  TraceConsumer() = default;
  virtual ~TraceConsumer() = default;
  TraceConsumer(const TraceConsumer&) = delete;
  TraceConsumer& operator=(const TraceConsumer&) = delete;

  virtual void process(void* data, size_t size) final {
    auto& flaggers = ThreadLocalApiCallFlaggers::getInstance();
    flaggers.setApiCallInProgress();
    processImpl(data, size);
    flaggers.resetApiCallInProgress();
  }
  virtual void processImpl(void* data, size_t size) = 0;
};

class TraceProducer {
 public:
  TraceProducer() = default;
  virtual ~TraceProducer() = default;
  TraceProducer(const TraceProducer&) = delete;
  TraceProducer& operator=(const TraceProducer&) = delete;

  virtual void apiHook(CUcontext ctx, int is_exit, const char* name,
                       void* params, CUresult* pStatus) final {
    auto& flaggers = ThreadLocalApiCallFlaggers::getInstance();
    if (flaggers.isApiCallInProgress()) {
      // Nested API call detected, skip tracing to avoid recursion
      return;
    }
    flaggers.setApiCallInProgress();
    if (is_exit) {
      onEndHook(ctx, name, params, pStatus);
    } else {
      onStartHook(ctx, name, params, pStatus);
    }
    flaggers.resetApiCallInProgress();
  }

 private:
  virtual void onStartHook(CUcontext ctx, const char* name, void* params,
                           CUresult* pStatus) = 0;
  virtual void onEndHook(CUcontext ctx, const char* name, void* params,
                         CUresult* pStatus) = 0;
};

};  // namespace notrace

#endif  // __NOTRACE_HANDLERS_BASE_HANDLER_H__
