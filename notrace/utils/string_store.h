#ifndef NOTRACE_UTILS_STRING_STORE_H_
#define NOTRACE_UTILS_STRING_STORE_H_

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace notrace {

using StringId = uint64_t;
class StringStore {
 private:
  std::unordered_map<std::string, StringId> _map;
  std::vector<const std::string*> _reverseMap;
  std::shared_mutex _mutex;
  StringStore() { _reverseMap.reserve(1024); }
  ~StringStore() = default;
  StringStore(const StringStore&) = delete;
  StringStore& operator=(const StringStore&) = delete;

 public:
  static StringStore& getInstance() {
    static StringStore instance;
    return instance;
  }
  StringId getStringId(const std::string& str);
  const std::string& getStringFromId(StringId id);
  std::vector<std::string> getSnapshot();
};

}  // namespace notrace

#endif  // NOTRACE_UTILS_STRING_STORE_H_
