#ifndef NOTRACE_TRACKER_MEMORY_TRACKER_H__
#define NOTRACE_TRACKER_MEMORY_TRACKER_H__

#include <cassert>
#include <cstddef>
#include <unordered_map>

namespace notrace {
namespace memory_tracker {

class MemoryTracker {
 public:
  static MemoryTracker& getInstance() {
    static MemoryTracker instance;
    return instance;
  }

  bool exists(void* ptr);

  void recordAllocation(void* ptr, size_t size);

  void recordDeallocation(void* ptr);

  size_t getAllocationSize(void* ptr);

 private:
  MemoryTracker() = default;
  ~MemoryTracker();
  MemoryTracker(const MemoryTracker&) = delete;
  MemoryTracker& operator=(const MemoryTracker&) = delete;
  std::unordered_map<void*, size_t> pointerMap;
};

}  // namespace memory_tracker
}  // namespace notrace

#endif  // NOTRACE_TRACKER_MEMORY_TRACKER_H__
