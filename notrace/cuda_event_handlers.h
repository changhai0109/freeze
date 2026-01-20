#ifndef __NOTRACE_CUDA_EVENT_HANDLERS_H__
#define __NOTRACE_CUDA_EVENT_HANDLERS_H__

#include <unordered_map>
#include "nvbit.h"

namespace notrace {

using cuda_event_handler_t = void(CUcontext, int, const char*, void*,
                                  CUresult*);

void register_handlers();

void register_cuda_event_handler(nvbit_api_cuda_t api_id,
                                 cuda_event_handler_t* handler);
void unregister_cuda_event_handler(nvbit_api_cuda_t api_id);
cuda_event_handler_t* get_cuda_event_handler(nvbit_api_cuda_t api_id);

}  // namespace notrace

#endif  // __NOTRACE_CUDA_EVENT_HANDLERS_H__
