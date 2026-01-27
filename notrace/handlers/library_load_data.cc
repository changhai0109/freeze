#include "handlers/library_load_data.h"
#include "common.h"
#include "utils/mpsc_queue.h"

namespace notrace {
namespace library_load_data {
static thread_local MessageWritter messageWritter;
static thread_local LibraryLoadDataProducer libraryLoadDataProducer;

void LibraryLoadDataProducer::onStartHook(CUcontext ctx, const char* name,
                                          void* params, CUresult* pStatus) {
  // Reserve space in the queue
  auto* msg = messageWritter.reserve<LibraryLoadDataParams>(
      nvbit_api_cuda_t::API_CUDA_cuLibraryLoadData);

  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve message in LibraryLoadDataProducer");
    return;
  }

  cuLibraryLoadData_params* p =
      reinterpret_cast<cuLibraryLoadData_params*>(params);

  msg->messageType = MESSAGE_TYPE_START;

  messageWritter.commit();
}

void LibraryLoadDataProducer::onEndHook(CUcontext ctx, const char* name,
                                        void* params, CUresult* pStatus) {
  // Reserve space in the queue
  auto* msg = messageWritter.reserve<LibraryLoadDataParams>(
      nvbit_api_cuda_t::API_CUDA_cuLibraryLoadData);

  if (msg == nullptr) [[unlikely]] {
    assert(false && "Failed to reserve message in LibraryLoadDataProducer");
    return;
  }

  cuLibraryLoadData_params* p =
      reinterpret_cast<cuLibraryLoadData_params*>(params);

  msg->messageType = MESSAGE_TYPE_END;
  messageWritter.commit();
}

void LibraryLoadDataConsumer::processImpl(void* data, size_t size) {
  // Currently, we do not process the library load data messages.
  // This function is a placeholder for future implementation.
  LibraryLoadDataParams* params =
      reinterpret_cast<LibraryLoadDataParams*>(data);
  if (params->messageType == MESSAGE_TYPE_START) {
    // Handle start message
    printf("Library load data start message received.\n");
  } else if (params->messageType == MESSAGE_TYPE_END) {
    // Handle end message
    printf("Library load data end message received.\n");
  } else {
    // Unknown message type
  }
}

void libraryLoadDataHookWrapper(CUcontext ctx, int is_exit, const char* name,
                                void* params, CUresult* pStatus) {
  libraryLoadDataProducer.apiHook(ctx, is_exit, name, params, pStatus);
}

}  // namespace library_load_data
}  // namespace notrace
