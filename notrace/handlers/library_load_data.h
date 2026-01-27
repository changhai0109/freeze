#ifndef NOTRACE_HANDLER_LIBRARY_LODA_DATA_H
#define NOTRACE_HANDLER_LIBRARY_LODA_DATA_H

#include "common.h"
#include "handlers/base_handler.h"

namespace notrace {
namespace library_load_data {

const uint8_t MESSAGE_TYPE_START = 1;
const uint8_t MESSAGE_TYPE_END = 2;

#pragma pack(push, 1)
struct LibraryLoadDataParams {
  uint8_t messageType;
  CUlibrary* library;
  const void* image;
  unsigned int numOptions;
  CUjit_option* options;
  void** optionValues;
};
#pragma pack(pop)

class LibraryLoadDataProducer : public TraceProducer {
 private:
  void onStartHook(CUcontext ctx, const char* name, void* params,
                   CUresult* pStatus) override;

  void onEndHook(CUcontext ctx, const char* name, void* params,
                 CUresult* pStatus) override;
};

class LibraryLoadDataConsumer : public TraceConsumer {
 private:
  void processImpl(void* data, size_t size) override;
};

void libraryLoadDataHookWrapper(CUcontext ctx, int is_exit, const char* name,
                                void* params, CUresult* pStatus);

}  // namespace library_load_data
}  // namespace notrace

#endif  // NOTRACE_HANDLER_LIBRARY_LODA_DATA_H
