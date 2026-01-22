#ifndef NOTRACE_UTILS_API_CALL_FLAGGERS_H__
#define NOTRACE_UTILS_API_CALL_FLAGGERS_H__

#include <cstddef>
#include "common.h"

namespace notrace {

class ThreadLocalApiCallFlaggers {
 public:
  static ThreadLocalApiCallFlaggers& getInstance() {
    static thread_local ThreadLocalApiCallFlaggers instance;
    return instance;
  }

  void setApiCallInProgress() { depth++; }

  void resetApiCallInProgress() {
    if (depth > 0) {
      depth--;
    }
  }

  bool isApiCallInProgress() const { return depth > 0; }

 private:
  ThreadLocalApiCallFlaggers() { depth = 0u; };
  uint8_t depth;
};

}  // namespace notrace

#endif  // NOTRACE_UTILS_API_CALL_FLAGGERS_H__
