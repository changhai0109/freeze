#include "tracker/memory_tracker.h"
#include <cstdint>
#include <cstdio>

namespace notrace {
namespace memory_tracker {

bool MemoryTracker::exists(void* ptr) {
  return pointerMap.find(ptr) != pointerMap.end();
}

void MemoryTracker::recordAllocation(void* ptr, size_t size,
                                     Location location) {
  printf("Recording allocation: ptr=%p, size=%zu\n", ptr, size);
  if (exists(ptr)) {
    assert(false && "Double allocation detected for the same pointer");
    return;
  }
  pointerMap[ptr] = std::make_pair(size, location);
}

void MemoryTracker::recordDeallocation(void* ptr) {
  auto it = pointerMap.find(ptr);
  if (it == pointerMap.end()) {
    printf("Deallocation of untracked pointer: %p\n", ptr);
    fflush(stdout);
    assert(false && "Deallocation of untracked pointer detected");
    return;
  }
  pointerMap.erase(it);
}

size_t MemoryTracker::getAllocationSize(void* ptr) const {
  auto it = pointerMap.find(ptr);
  if (it != pointerMap.end()) {
    return it->second.first;
  }
  assert(false && "Pointer not found in memory tracker");
  return 0;
}

Location MemoryTracker::getAllocationLocation(void* ptr) const {
  auto it = pointerMap.find(ptr);
  if (it != pointerMap.end()) {
    return it->second.second;
  }
  assert(false && "Pointer not found in memory tracker");
  return Location::UNKNOWN;  // Default return to avoid compiler warning
}

MemoryTracker::~MemoryTracker() {
  if (!pointerMap.empty()) {
    printf("Memory leaks detected:\n");
    for (const auto& pair : pointerMap) {
      printf("  Leaked ptr=%p, size=%zu location=%u\n", pair.first,
             pair.second.first, static_cast<uint8_t>(pair.second.second));
    }
    fflush(stdout);
  }
}

}  // namespace memory_tracker
}  // namespace notrace
