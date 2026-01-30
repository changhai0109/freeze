#include "utils/stream_event_mapper.h"
#include <chrono>
#include "utils/cuda_safecall.h"
#include "utils/event_pool.h"

namespace notrace {

CudaEventPool& cudaEventPool = CudaEventPool::getInstance();
StreamEventMapper& streamEventMapper = StreamEventMapper::getInstance();

void StreamEventMapper::recordStreamStart(const CUstream& stream,
                                          std::thread::id tid) {
  if (stream == NULL) {
    // Check if already recorded for this thread
    if (default_stream_start_map.find(tid) != default_stream_start_map.end()) {
      return;  // Already recorded
    }
  } else {
    // Check if already recorded for this stream
    if (stream_start_map.find(stream) != stream_start_map.end()) {
      return;  // Already recorded
    }
  }
  cudaEvent_t startEvent = cudaEventPool.acquire();
  uint64_t timestamp = 0;  // Placeholder for actual timestamp if needed
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::chrono::system_clock::duration dur = now.time_since_epoch();
  std::chrono::nanoseconds ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(dur);
  timestamp = static_cast<uint64_t>(ns.count());

  CUDA_SAFECALL(cudaEventRecordWithFlags(startEvent, stream, cudaEventDefault));

  if (stream == NULL) {
    default_stream_start_map[tid] = std::make_pair(startEvent, timestamp);
    return;
  }

  stream_start_map[stream] = std::make_pair(startEvent, timestamp);
}

inline const std::pair<cudaEvent_t, uint64_t>&
StreamEventMapper::getStreamStartEvent(const CUstream& stream,
                                       std::thread::id tid) const {
  if (stream == NULL) {
    auto it = default_stream_start_map.find(tid);
    if (it == default_stream_start_map.end()) {
      assert(false && "Default stream start event not found for thread");
      throw std::runtime_error(
          "Default stream start event not found for thread");
    }
    return it->second;
  }
  auto it = stream_start_map.find(stream);
  if (it == stream_start_map.end()) {
    assert(false && "Stream start event not found");
    throw std::runtime_error("Stream start event not found");
  }
  return it->second;
}

uint64_t StreamEventMapper::getStreamTimestamp(
    const CUstream& stream, const std::thread::id tid,
    const cudaEvent_t& endEvent) const {
  cudaEvent_t startEvent;
  uint64_t baseTimestamp;
  if (stream == NULL) {
    auto it = default_stream_start_map.find(tid);
    if (it == default_stream_start_map.end()) {
      assert(false && "Default stream start event not found for thread");
      throw std::runtime_error(
          "Default stream start event not found for thread");
    }
    startEvent = it->second.first;
    baseTimestamp = it->second.second;
  } else {
    auto it = stream_start_map.find(stream);
    if (it == stream_start_map.end()) {
      assert(false && "Stream start event not found");
      throw std::runtime_error("Stream start event not found");
    }
    startEvent = it->second.first;
    baseTimestamp = it->second.second;
  }
  float durationMs = 0.0f;
  CUDA_SAFECALL(cudaEventElapsedTime(&durationMs, startEvent, endEvent));
  return static_cast<uint64_t>(durationMs * 1e6) + baseTimestamp;  // ns
}

StreamEventMapper::~StreamEventMapper() {
  // Release all acquired events
  for (auto& pair : stream_start_map) {
    cudaEventPool.release(pair.second.first);
  }
  stream_start_map.clear();

  for (auto& pair : default_stream_start_map) {
    cudaEventPool.release(pair.second.first);
  }
  default_stream_start_map.clear();
}
}  // namespace notrace
