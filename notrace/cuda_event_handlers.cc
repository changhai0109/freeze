#include "cuda_event_handlers.h"
#include "handlers/ctx_get_current.h"
#include "handlers/default_handler.h"
#include "handlers/launch_kernel.h"
#include "handlers/library_load_data.h"
#include "handlers/mem_alloc.h"
#include "handlers/mem_free.h"
#include "handlers/stream_create.h"
#include "nvbit.h"

using cuda_event_handler_t = notrace::cuda_event_handler_t;

// static std::unordered_map<nvbit_api_cuda_t, cuda_event_handler_t*>
//     cuda_event_handlers;
static std::vector<cuda_event_handler_t*> cuda_event_handlers(128, nullptr);

void notrace::register_cuda_event_handler(
    nvbit_api_cuda_t cbid, notrace::cuda_event_handler_t* handler) {
  if (cbid >= cuda_event_handlers.size()) {
    cuda_event_handlers.resize(cbid * 2, nullptr);
  }
  cuda_event_handlers[cbid] = handler;
}

void notrace::unregister_cuda_event_handler(nvbit_api_cuda_t cbid) {
  cuda_event_handlers[cbid] = nullptr;
}

notrace::cuda_event_handler_t* notrace::get_cuda_event_handler(
    nvbit_api_cuda_t cbid) {
  if (cbid < cuda_event_handlers.size() &&
      cuda_event_handlers[cbid] != nullptr) {
    return cuda_event_handlers[cbid];
  } else {
    if constexpr (!notrace::default_handler::ENABLE_DEFAULT_HANDLER)
      return nullptr;
    notrace::default_handler::setCBID(
        cbid);  // set the cbid for default handler
    return notrace::default_handler::defaultHandlerHookWrapper;
  }
}

void notrace::register_handlers() {
  // register_cuda_event_handler(API_CUDA_cuLaunchKernel,
  // &notrace::handleKernelLaunch);
  register_cuda_event_handler(API_CUDA_cuLaunchKernel,
                              notrace::kernel_launch::launchKernelHookWrapper);
  // register_cuda_event_handler(API_CUDA_cuMemAlloc_v2,
  //                             notrace::mem_alloc::memAllocHookWrapper);
  // register_cuda_event_handler(API_CUDA_cuMemAlloc,
  //                             notrace::mem_alloc::memAllocHookWrapper);
  // register_cuda_event_handler(API_CUDA_cuMemFree_v2,
  //                             notrace::mem_free::memFreeHookWrapper);
  // register_cuda_event_handler(API_CUDA_cuMemFree,
  //                             notrace::mem_free::memFreeHookWrapper);
  // register_cuda_event_handler(
  //     API_CUDA_cuLibraryLoadData,
  //     notrace::library_load_data::libraryLoadDataHookWrapper);
  // register_cuda_event_handler(
  //     API_CUDA_cuStreamCreate,
  //     notrace::stream_create::streamCreateProducerWrapper);
  // register_cuda_event_handler(
  //     API_CUDA_cuStreamCreateWithPriority,
  //     notrace::stream_create::streamCreateWithPriorityProducerWrapper);
  // register_cuda_event_handler(
  //     API_CUDA_cuCtxGetCurrent,
  //     notrace::ctx_get_current::ctxGetCurrentProducerWrapper);
}