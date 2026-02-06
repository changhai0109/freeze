#include "tracker/memory_tracker.h"
#include <cstdint>
#include <cstdio>
#include "common.h"

namespace notrace {
namespace memory_tracker {

bool MemoryTracker::exists(void* ptr) {
  if constexpr (!ENABLE_MEMORY_TRACKING) {
    return true;
  }
  return activePointerMap.find(ptr) != activePointerMap.end();
}

void MemoryTracker::recordAllocation(void* ptr, size_t size,
                                     Location location) {
  if constexpr (!ENABLE_MEMORY_TRACKING) {
    return;
  }
  // printf("Recording allocation: ptr=%p, size=%zu\n", ptr, size);
  if (exists(ptr)) {
    assert(false && "Double allocation detected for the same pointer");
    return;
  }
  activePointerMap[ptr] = std::make_pair(size, location);
}

void MemoryTracker::recordDeallocation(void* ptr) {
  if constexpr (!ENABLE_MEMORY_TRACKING) {
    return;
  }
  auto it = activePointerMap.find(ptr);
  if (it == activePointerMap.end()) {
    printf("Deallocation of untracked pointer: %p\n", ptr);
    fflush(stdout);
    assert(false && "Deallocation of untracked pointer detected");
    return;
  }
  freedPointerMap[ptr] = it->second;
  activePointerMap.erase(it);
}

size_t MemoryTracker::getAllocationSize(void* ptr) const {
  if constexpr (!ENABLE_MEMORY_TRACKING) {
    return 0;
  }
  auto it = activePointerMap.find(ptr);
  if (it != activePointerMap.end()) {
    return it->second.first;
  }
  assert(false && "Pointer not found in memory tracker");
  return 0;
}

Location MemoryTracker::getAllocationLocation(void* ptr) const {
  if constexpr (!ENABLE_MEMORY_TRACKING) {
    return Location::UNKNOWN;
  }
  auto it = activePointerMap.find(ptr);
  if (it != activePointerMap.end()) {
    return it->second.second;
  }
  assert(false && "Pointer not found in memory tracker");
  return Location::UNKNOWN;  // Default return to avoid compiler warning
}

MemoryTracker::~MemoryTracker() {
  if (!activePointerMap.empty()) {
    printf("Memory leaks detected:\n");
    for (const auto& pair : activePointerMap) {
      printf("  Leaked ptr=%p, size=%zu location=%u\n", pair.first,
             pair.second.first, static_cast<uint8_t>(pair.second.second));
    }
    fflush(stdout);
  }
  if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
    if (!freedPointerMap.empty()) {
      printf("Freed pointers record:\n");
      for (const auto& pair : freedPointerMap) {
        printf("  Freed ptr=%p, size=%zu location=%u\n", pair.first,
               pair.second.first, static_cast<uint8_t>(pair.second.second));
      }
      fflush(stdout);
    }
}

}  // namespace memory_tracker
}  // namespace notrace
