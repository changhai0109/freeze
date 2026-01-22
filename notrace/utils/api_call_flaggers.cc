#include "utils/api_call_flaggers.h"

namespace notrace {

void ThreadLocalApiCallFlaggers::resetApiCallInProgress() {
  if (depth > 0) {
    depth--;
  }
}

void ThreadLocalApiCallFlaggers::setApiCallInProgress() {
  depth++;
}

bool ThreadLocalApiCallFlaggers::isApiCallInProgress() const {
  return depth > 0;
}

}  // namespace notrace
