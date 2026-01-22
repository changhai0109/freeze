#ifndef NOTRACE_HANDLERS_MEM_ALLOC_H__
#define NOTRACE_HANDLERS_MEM_ALLOC_H__

#include <cstddef>
#include "common.h"
#include "handlers/base_handler.h"

namespace notrace {
namespace mem_alloc {

#pragma pack(push, 1)
struct MemAllocRecord {
  CUdeviceptr ptr;
  size_t size;
};
#pragma pack(pop)

class MemAllocProducer : public TraceProducer {
 public:
  MemAllocProducer() = default;
  ~MemAllocProducer() override = default;

 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;
  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class MemAllocConsumer : public TraceConsumer {
 public:
  MemAllocConsumer() = default;
  ~MemAllocConsumer() override = default;

 private:
  void processImpl(void* data, size_t size) override;
};

void memAllocHookWrapper(CUcontext ctx, int is_exit, const char* name,
                         void* params, CUresult* pStatus);

}  // namespace mem_alloc
}  // namespace notrace

#endif  // NOTRACE_HANDLERS_MEM_ALLOC_H__
