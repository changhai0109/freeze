#ifndef NOTRACE_UTILS_STRING_STORE_H_
#define NOTRACE_UTILS_STRING_STORE_H_

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace notrace {

class StringStore {
 private:
  std::unordered_map<std::string, uint64_t> _map;
  std::vector<const std::string*> _reverseMap;
  std::shared_mutex _mutex;

 public:
  StringStore() { _reverseMap.reserve(1024); }

  uint64_t getStringId(const std::string& str);
  const std::string& getStringFromId(uint64_t id);
  std::vector<std::string> getSnapshot();
};

extern StringStore stringStore;

}  // namespace notrace

#endif  // NOTRACE_UTILS_STRING_STORE_H_
