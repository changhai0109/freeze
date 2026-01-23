#include "tracker/memory_tracker.h"
#include <cstdio>

namespace notrace {
namespace memory_tracker {

bool MemoryTracker::exists(void* ptr) {
  return pointerMap.find(ptr) != pointerMap.end();
}

void MemoryTracker::recordAllocation(void* ptr, size_t size) {
  printf("Recording allocation: ptr=%p, size=%zu\n", ptr, size);
  if (exists(ptr)) {
    assert(false && "Double allocation detected for the same pointer");
    return;
  }
  pointerMap[ptr] = size;
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

size_t MemoryTracker::getAllocationSize(void* ptr) {
  auto it = pointerMap.find(ptr);
  if (it != pointerMap.end()) {
    return it->second;
  }
  assert(false && "Pointer not found in memory tracker");
  return 0;
}

MemoryTracker::~MemoryTracker() {
  if (!pointerMap.empty()) {
    printf("Memory leaks detected:\n");
    for (const auto& pair : pointerMap) {
      printf("  Leaked ptr=%p, size=%zu\n", pair.first, pair.second);
    }
    fflush(stdout);
  }
}

}  // namespace memory_tracker
}  // namespace notrace
