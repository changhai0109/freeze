#ifndef NOTRACE_HANDLERS_STREAM_CREATE_H
#define NOTRACE_HANDLERS_STREAM_CREATE_H

#include <thread>
#include "handlers/base_handler.h"

namespace notrace {
namespace stream_create {

constexpr int32_t DEFAULT_PRIORITY = INT32_MIN;
#pragma pack(push, 1)
typedef struct {
  CUstream stream;
  uint32_t flags;
  int32_t priority;
  std::thread::id tid;
} StreamCreateMsg;
#pragma pack(pop)

class StreamCreateProducer : public TraceProducer {
 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;
  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class StreamCreateWithPriorityProducer : public TraceProducer {
 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;
  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class StreamCreateConsumer : public TraceConsumer {
 private:
  void processImpl(void* data, size_t size) override;
};

void streamCreateProducerWrapper(CUcontext ctx, int is_exit, const char* name,
                                 void* params, CUresult* pStatus);
void streamCreateWithPriorityProducerWrapper(CUcontext ctx, int is_exit,
                                             const char* name, void* params,
                                             CUresult* pStatus);

}  // namespace stream_create
}  // namespace notrace

#endif  // NOTRACE_HANDLERS_STREAM_CREATE_H
