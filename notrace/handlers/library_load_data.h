#ifndef NOTRACE_HANDLER_LIBRARY_LODA_DATA_H
#define NOTRACE_HANDLER_LIBRARY_LODA_DATA_H

#include "common.h"
#include "cuda.h"
#include "generated_cuda_meta.h"
#include "handlers/base_handler.h"
#include "utils/memory_inspector.h"

namespace notrace {
namespace library_load_data {

constexpr uint8_t RAW_DATA = 1;
constexpr uint8_t PARSED_DATA = 2;
#pragma pack(push, 1)
struct LibraryLoadDataRawMessage {
  uint8_t dataType = RAW_DATA;
  cuLibraryLoadData_params params;
};
#pragma pack(pop)

constexpr size_t MAX_JIT_OPTIONS = 8;
constexpr size_t MAX_JIT_OPTION_VALUE_BYTES = 256;
constexpr size_t MAX_LIBRARY_OPTIONS = 4;
constexpr size_t MAX_LIBRARY_OPTION_VALUE_BYTES = 8;

#pragma pack(push, 1)
struct LibraryLoadDataParsedMessage {
  uint8_t dataType = PARSED_DATA;
  bool hasRegion;
  MemoryRegion region;

  struct JitOption {
    CUjit_option option;
    uint8_t valueSize;
    uint16_t valueOffset;
  };
  JitOption jitOptions[MAX_JIT_OPTIONS];
  uint8_t jitOptionCount;
  uint8_t jitOptionValuesBuffer[MAX_JIT_OPTION_VALUE_BYTES];
  uint16_t jitOptionsBufferSizeUsed;

  size_t jitOptionsSize(const CUjit_option& opt) const {
    // just assume everything is 4 bytes for now
    return 4;
    // switch (opt) {
    //   case:
    //     return 1;
    //   case:
    //     return 2;
    //   case CU_JIT_MAX_REGISTERS:
    //   case CU_JIT_THREADS_PER_BLOCK:
    //   case CU_JIT_WALL_TIME:
    //   case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
    //   case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
    //   case CU_JIT_OPTIMIZATION_LEVEL:
    //   case CU_JIT_TARGET:
    //   case CU_JIT_FALLBACK_STRATEGY:
    //   case CU_JIT_GENERATE_DEBUG_INFO:
    //   case CU_JIT_LOG_VERBOSE:
    //   case CU_JIT_GENERATE_LINE_INFO:
    //   case CU_JIT_CACHE_MODE:
    //   case CU_JIT_NEW_SM3X_OPT:
    //   case CU_JIT_FAST_COMPILE:

    //     return 4;
    //   case:
    //     return 8;
    //   case CU_JIT_TARGET_FROM_CUCONTEXT:
    //     return 0;  // No value
    //   case CU_JIT_ERROR_LOG_BUFFER:
    //   case CU_JIT_INFO_LOG_BUFFER:
    //   case CU_JIT_GLOBAL_SYMBOL_NAMES:
    //   case CU_JIT_GLOBAL_SYMBOL_ADDRESSES:
    //     return 256;  // Variable size, handle separately
    // }
  }

  struct LibraryOption {
    CUlibraryOption option;
    uint8_t valueSize;
    uint16_t valueOffset;
  };
  LibraryOption libraryOptions[MAX_LIBRARY_OPTIONS];
  uint8_t libraryOptionCount;
  uint8_t libraryOptionValuesBuffer[MAX_LIBRARY_OPTION_VALUE_BYTES];
  uint16_t libraryOptionsBufferSizeUsed;

  size_t libraryOptionsSize(const CUlibraryOption& opt) const {
    // just assume everything is 4 bytes for now
    return 0;
  }
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
  void processRawMessage(void* data, size_t size);
  void processParsedMessage(void* data, size_t size);
  void processImpl(void* data, size_t size) override;
};

void libraryLoadDataHookWrapper(CUcontext ctx, int is_exit, const char* name,
                                void* params, CUresult* pStatus);

}  // namespace library_load_data
}  // namespace notrace

#endif  // NOTRACE_HANDLER_LIBRARY_LODA_DATA_H
