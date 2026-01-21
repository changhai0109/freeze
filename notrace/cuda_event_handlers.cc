#include "cuda_event_handlers.h"
#include "handlers/kernel_launch.h"
#include "nvbit.h"

using cuda_event_handler_t = notrace::cuda_event_handler_t;

static std::unordered_map<nvbit_api_cuda_t, cuda_event_handler_t*>
    cuda_event_handlers;

void notrace::register_cuda_event_handler(
    nvbit_api_cuda_t cbid, notrace::cuda_event_handler_t* handler) {
  cuda_event_handlers[cbid] = handler;
}

void notrace::unregister_cuda_event_handler(nvbit_api_cuda_t cbid) {
  cuda_event_handlers.erase(cbid);
}

notrace::cuda_event_handler_t* notrace::get_cuda_event_handler(
    nvbit_api_cuda_t cbid) {
  auto it = cuda_event_handlers.find(cbid);
  if (it != cuda_event_handlers.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

void notrace::register_handlers() {
  // register_cuda_event_handler(API_CUDA_cuLaunchKernel,
  // &notrace::handleKernelLaunch);
  register_cuda_event_handler(API_CUDA_cuLaunchKernel,
                              notrace::kernel_launch::kernelLaunchHookWrapper);
}