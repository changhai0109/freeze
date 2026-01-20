#ifndef _NOTRACE_UTILS_CUDA_SAFECALL_H__
#define _NOTRACE_UTILS_CUDA_SAFECALL_H__
#include <cstdio>
#include <cstdlib>

#define CUDA_SAFECALL(call)                                                \
  do {                                                                     \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
      fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%d)\n", __FILE__, __LINE__, \
              cudaGetErrorString(_err), (int)_err);                        \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

#endif
