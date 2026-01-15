#include "cuda_event_handlers.h"
#include "nvbit.h"
#include "nvbit_tool.h"

void nvbit_at_init() {
  printf("Tool is being loaded\n");
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
  notrace::cuda_event_handler_t* handler =
      notrace::get_cuda_event_handler(cbid);
  if (handler != nullptr)
    return handler(ctx, is_exit, name, params, pStatus);

  // default action: print event info
  printf("CUDA API %s %s\n", is_exit ? "EXIT" : "ENTER", name);
}

void nvbit_at_term() {
  printf("Tool is being unloaded\n");
  fflush(stdout);
}

// Fake function to force nvbit is linked and not optimized out
void __fake_nvbit_loader(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);
}
