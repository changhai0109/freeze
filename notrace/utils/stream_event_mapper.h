#ifndef NOTRACE_UTILS_STREAM_EVENT_MAPPER_H
#define NOTRACE_UTILS_STREAM_EVENT_MAPPER_H
#include <cuda_runtime.h>
#include <thread>
#include <unordered_map>
#include "common.h"

namespace notrace {

class StreamEventMapper {
 public:
  static StreamEventMapper& getInstance() {
    static StreamEventMapper instance;
    return instance;
  }

  void recordStreamStart(const CUstream& stream, const std::thread::id tid);
  const std::pair<cudaEvent_t, uint64_t>& getStreamStartEvent(
      const CUstream& stream, const std::thread::id tid) const;

  uint64_t getStreamTimestamp(const CUstream& stream, const std::thread::id tid,
                              const cudaEvent_t& event);

 private:
  StreamEventMapper() = default;
  ~StreamEventMapper();

  std::unordered_map<CUstream, std::pair<cudaEvent_t, uint64_t>>
      stream_start_map;
  std::unordered_map<std::thread::id, std::pair<cudaEvent_t, uint64_t>>
      default_stream_start_map;
};

}  // namespace notrace

#endif  // NOTRACE_UTILS_STREAM_EVENT_MAPPER_H
