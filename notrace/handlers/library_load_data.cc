#include "handlers/library_load_data.h"
#include <cassert>
#include <cstring>
#include <optional>
#include "common.h"
#include "generated_cuda_meta.h"
#include "utils/memory_inspector.h"
#include "utils/mpsc_queue.h"

namespace notrace {
namespace library_load_data {
static thread_local MessageWritter messageWritter;
static thread_local LibraryLoadDataProducer libraryLoadDataProducer;

void LibraryLoadDataProducer::onStartHook(CUcontext ctx, const char* name,
                                          void* params, CUresult* pStatus) {
  // do nothing on start
}

void LibraryLoadDataProducer::onEndHook(CUcontext ctx, const char* name,
                                        void* params, CUresult* pStatus) {
  LibraryLoadDataParsedMessage* parsedMsg =
      messageWritter.reserve<LibraryLoadDataParsedMessage>(
          nvbit_api_cuda_t::API_CUDA_cuLibraryLoadData);

  if (parsedMsg == nullptr) [[unlikely]] {
    assert(false &&
           "Failed to reserve parsed message in LibraryLoadDataConsumer");
    return;
  }

  cuLibraryLoadData_params* p =
      reinterpret_cast<cuLibraryLoadData_params*>(params);

  parsedMsg->dataType = PARSED_DATA;
  std::optional<MemoryRegion> region_opt = find_address_in_maps(p->code);
  if (region_opt.has_value()) {
    MemoryRegion region = region_opt.value();
    parsedMsg->hasRegion = true;
    parsedMsg->region = region;
  } else {
    parsedMsg->hasRegion = false;
  }
  auto& jitValueOffset = parsedMsg->jitOptionsBufferSizeUsed;
  for (size_t i = 0; i < p->numJitOptions; i++) {
    if (i >= MAX_JIT_OPTIONS) [[unlikely]] {
      assert(false && "Exceeded MAX_JIT_OPTIONS in LibraryLoadDataConsumer");
      break;
    }

    parsedMsg->jitOptions[i].option = p->jitOptions[i];
    size_t optionSize = parsedMsg->jitOptionsSize(p->jitOptions[i]);
    if (jitValueOffset + optionSize > MAX_JIT_OPTION_VALUE_BYTES) [[unlikely]] {
      assert(false &&
             "Exceeded MAX_JIT_OPTION_VALUE_BYTES in LibraryLoadDataConsumer");
      break;
    }
    parsedMsg->jitOptions[i].valueSize = static_cast<uint8_t>(optionSize);
    parsedMsg->jitOptions[i].valueOffset =
        static_cast<uint16_t>(jitValueOffset);
    // Copy the option value
    if (optionSize > 0)
      std::memcpy(&parsedMsg->jitOptionValuesBuffer[jitValueOffset],
                  reinterpret_cast<uint8_t*>(p->jitOptionsValues[i]),
                  optionSize);
    jitValueOffset += optionSize;
    parsedMsg->jitOptionCount++;
  }

  auto& libValueOffset = parsedMsg->libraryOptionsBufferSizeUsed;
  for (size_t i = 0; i < p->numLibraryOptions; i++)
    [[unlikely]] {
      if (i >= MAX_LIBRARY_OPTIONS) {
        assert(false &&
               "Exceeded MAX_LIBRARY_OPTIONS in LibraryLoadDataConsumer");
        break;
      }
      parsedMsg->libraryOptions[i].option = p->libraryOptions[i];
      size_t optionSize = parsedMsg->libraryOptionsSize(p->libraryOptions[i]);
      if (libValueOffset + optionSize > MAX_LIBRARY_OPTION_VALUE_BYTES)
          [[unlikely]] {
        assert(false &&
               "Exceeded MAX_LIBRARY_OPTION_VALUE_BYTES in "
               "LibraryLoadDataConsumer");
        break;
      }
      parsedMsg->libraryOptions[i].valueSize = static_cast<uint8_t>(optionSize);
      parsedMsg->libraryOptions[i].valueOffset =
          static_cast<uint16_t>(libValueOffset);
      // Copy the option value
      if (optionSize > 0)
        std::memcpy(&parsedMsg->libraryOptionValuesBuffer[libValueOffset],
                    reinterpret_cast<uint8_t*>(p->libraryOptionValues[i]),
                    optionSize);
      libValueOffset += optionSize;
      parsedMsg->libraryOptionCount++;
    }
  messageWritter.commit();
}

void LibraryLoadDataConsumer::processRawMessage(void* data, size_t size) {
  // will never be called because no raw messages are produced currently
  LibraryLoadDataRawMessage* rawMsg =
      reinterpret_cast<LibraryLoadDataRawMessage*>(data);
  cuLibraryLoadData_params params = rawMsg->params;

  LibraryLoadDataParsedMessage* parsedMsg =
      messageWritter.reserve<LibraryLoadDataParsedMessage>(
          nvbit_api_cuda_t::API_CUDA_cuLibraryLoadData);

  if (parsedMsg == nullptr) [[unlikely]] {
    assert(false &&
           "Failed to reserve parsed message in LibraryLoadDataConsumer");
    return;
  }

  parsedMsg->dataType = PARSED_DATA;
  std::optional<MemoryRegion> region_opt = find_address_in_maps(params.code);
  if (region_opt.has_value()) {
    MemoryRegion region = region_opt.value();
    parsedMsg->hasRegion = true;
    parsedMsg->region = region;
  } else {
    parsedMsg->hasRegion = false;
  }
  auto& jitValueOffset = parsedMsg->jitOptionsBufferSizeUsed;
  for (size_t i = 0; i < params.numJitOptions; i++) {
    if (i >= MAX_JIT_OPTIONS) [[unlikely]] {
      assert(false && "Exceeded MAX_JIT_OPTIONS in LibraryLoadDataConsumer");
      break;
    }

    parsedMsg->jitOptions[i].option = params.jitOptions[i];
    size_t optionSize = parsedMsg->jitOptionsSize(params.jitOptions[i]);
    if (jitValueOffset + optionSize > MAX_JIT_OPTION_VALUE_BYTES) [[unlikely]] {
      assert(false &&
             "Exceeded MAX_JIT_OPTION_VALUE_BYTES in LibraryLoadDataConsumer");
      break;
    }
    parsedMsg->jitOptions[i].valueSize = static_cast<uint8_t>(optionSize);
    parsedMsg->jitOptions[i].valueOffset =
        static_cast<uint16_t>(jitValueOffset);
    // Copy the option value
    std::memcpy(&parsedMsg->jitOptionValuesBuffer[jitValueOffset],
                reinterpret_cast<uint8_t*>(params.jitOptionsValues[i]),
                optionSize);
    jitValueOffset += optionSize;
    parsedMsg->jitOptionCount++;
  }

  auto& libValueOffset = parsedMsg->libraryOptionsBufferSizeUsed;
  for (size_t i = 0; i < params.numLibraryOptions; i++)
    [[unlikely]] {
      if (i >= MAX_LIBRARY_OPTIONS) {
        assert(false &&
               "Exceeded MAX_LIBRARY_OPTIONS in LibraryLoadDataConsumer");
        break;
      }
      parsedMsg->libraryOptions[i].option = params.libraryOptions[i];
      size_t optionSize =
          parsedMsg->libraryOptionsSize(params.libraryOptions[i]);
      if (libValueOffset + optionSize > MAX_LIBRARY_OPTION_VALUE_BYTES)
          [[unlikely]] {
        assert(false &&
               "Exceeded MAX_LIBRARY_OPTION_VALUE_BYTES in "
               "LibraryLoadDataConsumer");
        break;
      }
      parsedMsg->libraryOptions[i].valueSize = static_cast<uint8_t>(optionSize);
      parsedMsg->libraryOptions[i].valueOffset =
          static_cast<uint16_t>(libValueOffset);
      // Copy the option value
      std::memcpy(&parsedMsg->libraryOptionValuesBuffer[libValueOffset],
                  reinterpret_cast<uint8_t*>(params.libraryOptionValues[i]),
                  optionSize);
      libValueOffset += optionSize;
      parsedMsg->libraryOptionCount++;
    }
  messageWritter.commit();
}

void LibraryLoadDataConsumer::processParsedMessage(void* data, size_t size) {
  // Currently, we do not process the parsed library load data messages.
  // This function is a placeholder for future implementation.
  LibraryLoadDataParsedMessage* parsedMsg =
      reinterpret_cast<LibraryLoadDataParsedMessage*>(data);
  printf(
      "LibraryLoadDataParsedMessage: numJitOptions=%d, numLibraryOptions=%d\t",
      parsedMsg->jitOptionCount, parsedMsg->libraryOptionCount);
  if (parsedMsg->hasRegion) {
    printf("Mapped Region: ");
    parsedMsg->region.print();
  }
  printf("\n");
}

void LibraryLoadDataConsumer::processImpl(void* data, size_t size) {
  // Currently, we do not process the library load data messages.
  // This function is a placeholder for future implementation.
  if (size < 1) {
    return;
  }
  uint8_t dataType = *(reinterpret_cast<uint8_t*>(data));
  if (dataType == RAW_DATA) {
    processRawMessage(data, size);
  } else if (dataType == PARSED_DATA) {
    processParsedMessage(data, size);
  } else {
    assert(false && "Unknown data type in LibraryLoadDataConsumer");
  }
}

void libraryLoadDataHookWrapper(CUcontext ctx, int is_exit, const char* name,
                                void* params, CUresult* pStatus) {
  libraryLoadDataProducer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace library_load_data
}  // namespace notrace
