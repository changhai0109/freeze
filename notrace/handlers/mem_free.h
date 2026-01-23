#ifndef NOTRACE_HANDLERS_MEM_FREE_H
#define NOTRACE_HANDLERS_MEM_FREE_H

#include "common.h"
#include "handlers/base_handler.h"

namespace notrace {
namespace mem_free {
#pragma pack(push, 1)
struct MemFreeRecord {
  CUdeviceptr ptr;
};
#pragma pack(pop)

class MemFreeProducer : public TraceProducer {
 public:
  MemFreeProducer() = default;
  ~MemFreeProducer() override = default;

 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;
  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class MemFreeConsumer : public TraceConsumer {
 public:
  MemFreeConsumer() = default;
  ~MemFreeConsumer() override = default;

 private:
  void processImpl(void* data, size_t size) override;
};

void memFreeHookWrapper(CUcontext ctx, int is_exit, const char* name,
                        void* params, CUresult* pStatus);

}  // namespace mem_free
}  // namespace notrace

#endif  // NOTRACE_HANDLERS_MEM_FREE_H
