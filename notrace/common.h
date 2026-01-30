#ifndef __NOTRACE_COMMON_H__
#define __NOTRACE_COMMON_H__

#include "cuda.h"
#if CUDA_VERSION >= 12000
// CUdeviceptr_v1 was removed in CUDA 12, but NVBit needs it.
// We define it as CUdeviceptr (unsigned long long)
typedef CUdeviceptr CUdeviceptr_v1;
typedef CUDA_MEMCPY2D CUDA_MEMCPY2D_v1;
typedef CUDA_MEMCPY3D CUDA_MEMCPY3D_v1;
typedef CUDA_ARRAY_DESCRIPTOR CUDA_ARRAY_DESCRIPTOR_v1;
typedef CUDA_ARRAY3D_DESCRIPTOR CUDA_ARRAY3D_DESCRIPTOR_v1;

#endif

#include "nvbit.h"

namespace notrace {
namespace debug {
inline constexpr bool ENABLE_DEBUG_LOGS = false;
inline constexpr bool ENABLE_MPSC_DEBUG_LOGS = false && ENABLE_DEBUG_LOGS;
}  // namespace debug
}  // namespace notrace

#endif
