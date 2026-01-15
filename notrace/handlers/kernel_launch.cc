#include "handlers/kernel_launch.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include <atomic>
#include <unordered_map>
#include <vector>

namespace notrace {

std::atomic<uint64_t> global_launch_id{0};

thread_local std::vector<notrace::kernel_launch_info_t>
    thread_kernel_launch_info;

thread_local std::unordered_map<CUstream, std::vector<pending_kernel_t>>
    pending_kernels;

thread_local std::unordered_map<CUstream, cuda_stream_clock_t>
    initialized_stream_clocks;

thread_local CudaEventPool cuda_event_pool;

thread_local std::unordered_map<uint64_t, cuda_event_pair_t> pending_events;

inline uint64_t next_launch_id() {
  return global_launch_id.fetch_add(1, std::memory_order_relaxed);
}

void handle_kernel_launch(CUcontext ctx, int device_id, const char* kernel_name,
                          void* kernel_params, CUresult* pResult) {}

inline uint64_t get_host_time_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

uint64_t get_event_timestamp(CUstream steam, cudaEvent_t event) {
  if (initialized_stream_clocks.find(steam) ==
      initialized_stream_clocks.end()) {
    cuda_stream_clock_t stream_clock;
    cudaEventCreateWithFlags(&stream_clock.t0, cudaEventDefault);
    cudaEventRecord(stream_clock.t0, (cudaStream_t)steam);
    stream_clock.host_t0_ns = get_host_time_ns();
    initialized_stream_clocks[steam] = stream_clock;
  }

  cuda_stream_clock_t& stream_clock = initialized_stream_clocks[steam];
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, stream_clock.t0, event);
  uint64_t ns = static_cast<uint64_t>(ms * 1e6);
  return stream_clock.host_t0_ns + ns;
}

void on_kernel_enter(CUcontext ctx, CUstream stream, const char* kernel_name,
                     void** kernel_params) {
  kernel_launch_info_t rec{};

  rec.launch_id = next_launch_id();

  auto event_pair = cuda_event_pool.acquire();
  cudaEventRecord(event_pair.start, (cudaStream_t)stream);
  pending_events[rec.launch_id] = event_pair;

  rec.stream = stream;
  rec.context = ctx;
  rec.kernel_name = kernel_name;
  cuLaunchKernel_params_st* params =
      (cuLaunchKernel_params_st*)(*kernel_params);

  rec.grid_x = params->gridDimX;
  rec.grid_y = params->gridDimY;
  rec.grid_z = params->gridDimZ;
  rec.block_x = params->blockDimX;
  rec.block_y = params->blockDimY;
  rec.block_z = params->blockDimZ;
  rec.shared_mem_bytes = params->sharedMemBytes;
  rec.gpu_start_cycles = get_event_timestamp(stream, event_pair.start);

  size_t idx = thread_kernel_launch_info.size();
  thread_kernel_launch_info.push_back(rec);

  pending_kernels[stream].push_back({rec.launch_id, idx});
}

static cuda_event_pair_t create_event_pair() {
  cuda_event_pair_t pair;
  cudaEventCreateWithFlags(&pair.start, cudaEventDefault);
  cudaEventCreateWithFlags(&pair.end, cudaEventDefault);
  return pair;
}

cuda_event_pair_t CudaEventPool::acquire() {
  if (pool_.empty()) {
    return create_event_pair();
  } else {
    cuda_event_pair_t pair = pool_.back();
    pool_.pop_back();
    return pair;
  }
}

void CudaEventPool::release(cuda_event_pair_t pair) {
  pool_.push_back(pair);
}

CudaEventPool::~CudaEventPool() {
  for (auto& pair : pool_) {
    cudaEventDestroy(pair.start);
    cudaEventDestroy(pair.end);
  }
}

void CUDART_CB kernel_done_callback(void* data) {
  auto* p = static_cast<KernelDonePayload*>(data);
  p->rec->gpu_start_cycles =
      get_event_timestamp(p->rec->stream, p->events.start);
  p->rec->gpu_end_cycles = get_event_timestamp(p->rec->stream, p->events.end);

  cuda_event_pool.release(p->events);
  delete p;
}

void on_kernel_exit(CUcontext ctx, CUstream stream, const char* kernel_name,
                    void** kernel_params, CUresult* pResult) {
  auto& stack = pending_kernels[stream];
  auto pending = stack.back();
  stack.pop_back();
  auto& rec = thread_kernel_launch_info[pending.record_index];
  auto it = pending_events.find(rec.launch_id);
  assert(it != pending_events.end());
  auto event_pair = it->second;

  auto payload = new KernelDonePayload{
      .events = event_pair,
      .rec = &thread_kernel_launch_info[pending.record_index],
  };

  cudaEventRecord(event_pair.end, (cudaStream_t)stream);
  cudaLaunchHostFunc(stream, kernel_done_callback, payload);

  pending_events.erase(it);
}

}  // namespace notrace
