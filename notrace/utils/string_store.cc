#include "utils/string_store.h"

namespace notrace {

StringId StringStore::getStringId(const std::string& str) {
  {
    std::shared_lock<std::shared_mutex> readLock(_mutex);
    auto it = _map.find(str);
    if (it != _map.end()) {
      return it->second;
    }
  }

  std::unique_lock<std::shared_mutex> writeLock(_mutex);

  auto it = _map.find(str);
  if (it != _map.end()) {
    return it->second;
  }

  StringId newId = _reverseMap.size();
  auto result = _map.emplace(str, newId);

  const std::string* keyPtr = &(result.first->first);
  _reverseMap.push_back(keyPtr);

  return newId;
}

const std::string& StringStore::getStringFromId(StringId id) {
  std::shared_lock<std::shared_mutex> readLock(_mutex);
  if (id < _reverseMap.size()) {
    return *(_reverseMap[id]);
  }
  static const std::string empty = "<UNKNOWN>";
  return empty;
}

std::vector<std::string> StringStore::getSnapshot() {
  std::shared_lock<std::shared_mutex> readLock(_mutex);
  std::vector<std::string> snapshot;
  snapshot.reserve(_reverseMap.size());

  for (const auto* ptr : _reverseMap) {
    snapshot.push_back(*ptr);
  }
  return snapshot;
}

}  // namespace notrace
