// handlers/kernel_launch.cc
#include "handlers/kernel_launch.h"
#include <atomic>
#include "utils/mpsc_queue.h"

namespace notrace {
namespace kernel_launch {
std::atomic<uint64_t> launchIdCounter{0};
thread_local uint64_t currentLaunchId = 0;

inline uint64_t getAndSaveLaunchId() {
  currentLaunchId = launchIdCounter.fetch_add(1, std::memory_order_relaxed);
  return currentLaunchId;
}

inline uint64_t getCurrentLaunchId() {
  return currentLaunchId;
}

void onStart(CUcontext ctx, int device_id, const char* kernel_name,
             void* kernel_params, CUresult* pResult) {}

void onEnd(CUcontext ctx, int device_id, const char* kernel_name,
           void* kernel_params, CUresult* pResult) {}

};  // namespace kernel_launch
};  // namespace notrace
