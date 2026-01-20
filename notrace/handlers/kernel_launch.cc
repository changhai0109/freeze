// handlers/kernel_launch.cc
#include "handlers/kernel_launch.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>

#include <unistd.h>
#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "generated_cuda_meta.h"

#define CUDA_SAFECALL(call)                                                \
  do {                                                                     \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
      fprintf(stderr, "[CUDA ERROR] %s:%d: %s (%d)\n", __FILE__, __LINE__, \
              cudaGetErrorString(_err), (int)_err);                        \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

namespace notrace {

// -----------------------------
// Global / shared state
// -----------------------------
static std::atomic<uint64_t> globalLaunchId{0};

// Per-calling-thread record storage (stable addresses)
thread_local std::deque<KernelLaunchInfo> threadKernelLaunchInfo;

// NOTE: Callbacks run on CUDA internal worker threads, so anything they touch
// MUST NOT be thread_local. Keep shared structures global + protected by a mutex.
static std::mutex mutex;

// Stream clock anchors (per stream)
static std::unordered_map<CUstream, CudaStreamClock> streamClocks;

// Launch_id -> event pair used for that kernel
static std::unordered_map<uint64_t, CudaEventPair> pendingEvents;

// A global pool to reuse cuda events across threads/callbacks
static CudaEventPool eventPool;

// Track nested launches per-stream robustly without relying on thread_local stacks.
// We store direct pointers to records (stable due to deque).
struct PendingRecord {
  uint64_t launchId;
  KernelLaunchInfo* rec;
};
static std::unordered_map<CUstream, std::vector<PendingRecord>> pendingStack;
static std::vector<KernelLaunchInfo> completedKernelLaunches;

// -----------------------------
// Helpers
// -----------------------------
inline uint64_t next_launch_id() {
  return globalLaunchId.fetch_add(1, std::memory_order_relaxed);
}

static inline uint64_t get_host_time_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

// Must be called with g_mu held and stream clock already created.
static uint64_t get_event_timestamp_locked(CUstream stream, cudaEvent_t event) {
  auto it = streamClocks.find(stream);
  assert(it != streamClocks.end());

  CudaStreamClock& clk = it->second;

  float ms = 0.0f;
  // This returns milliseconds with ~0.5us resolution typically.
  CUDA_SAFECALL(cudaEventElapsedTime(&ms, clk.t0, event));

  uint64_t ns = static_cast<uint64_t>(ms * 1e6);
  return clk.host_t0_ns + ns;
}

static void ensure_stream_clock(CUstream stream) {
  std::lock_guard<std::mutex> lk(mutex);
  if (streamClocks.find(stream) != streamClocks.end())
    return;

  CudaStreamClock clk{};
  CUDA_SAFECALL(cudaEventCreateWithFlags(&clk.t0, cudaEventDefault));

  // Record anchor event into the same CUDA stream *before* we rely on it.
  CUDA_SAFECALL(cudaEventRecord(clk.t0, (cudaStream_t)stream));

  clk.host_t0_ns = get_host_time_ns();
  streamClocks.emplace(stream, clk);
}

// -----------------------------
// CudaEventPool impl
// -----------------------------
CudaEventPair CudaEventPool::createEventPair() {
  CudaEventPair pair{};

  CUDA_SAFECALL(cudaEventCreateWithFlags(&pair.start, cudaEventDefault));

  CUDA_SAFECALL(cudaEventCreateWithFlags(&pair.end, cudaEventDefault));

  return pair;
}

CudaEventPair CudaEventPool::acquire() {
  // Caller must synchronize externally if used across threads.
  if (pool_.empty())
    return createEventPair();

  CudaEventPair pair = pool_.back();
  pool_.pop_back();
  return pair;
}

void CudaEventPool::release(CudaEventPair pair) {
  pool_.push_back(pair);
}

CudaEventPool::~CudaEventPool() {
  for (auto& pair : pool_) {
    CUDA_SAFECALL(cudaEventDestroy(pair.start));
    CUDA_SAFECALL(cudaEventDestroy(pair.end));
  }
  pool_.clear();
}

// -----------------------------
// Kernel completion callback
// -----------------------------
static void CUDART_CB kernel_done_callback(void* data) {
  auto* p = static_cast<KernelDonePayload*>(data);
  if (!p || !p->rec) {
    delete p;
    return;
  }

  std::lock_guard<std::mutex> lk(mutex);

  float ms = 0.0f;
  p->rec->gpuStartCycles = 0;
  CUDA_SAFECALL(cudaEventElapsedTime(&ms, p->events.start, p->events.end));

  p->rec->gpuEndCycles = static_cast<uint64_t>(ms * 1e6);

  // Snapshot the completed record
  completedKernelLaunches.push_back(*p->rec);

  // Recycle events
  eventPool.release(p->events);
  delete p;
}

// -----------------------------
// Enter/exit hooks (called around cuLaunchKernel)
// -----------------------------
static void on_kernel_enter(CUcontext ctx, CUstream stream,
                            const char* kernel_name, void** kernel_params) {

  stream = reinterpret_cast<cuLaunchKernel_params_st*>(*kernel_params)->hStream;
  ensure_stream_clock(stream);

  KernelLaunchInfo rec{};
  rec.launch_id = next_launch_id();
  rec.stream = stream;
  rec.context = ctx;
  rec.kernel_name = kernel_name;

  // Extract launch params (this assumes you pass cuLaunchKernel params pointer)
  // If your hook wiring differs, adjust accordingly.
  if (kernel_params && *kernel_params) {
    auto* params = reinterpret_cast<cuLaunchKernel_params_st*>(*kernel_params);
    rec.gridX = params->gridDimX;
    rec.gridY = params->gridDimY;
    rec.gridZ = params->gridDimZ;
    rec.blockX = params->blockDimX;
    rec.blockY = params->blockDimY;
    rec.blockZ = params->blockDimZ;
    rec.sharedMemBytes = params->sharedMemBytes;
  }

  // Allocate stable storage for this record in this calling thread.
  threadKernelLaunchInfo.push_back(rec);
  KernelLaunchInfo* rec_ptr = &threadKernelLaunchInfo.back();

  // Acquire + record start event, then remember the pair by launch_id.
  CudaEventPair pair{};
  {
    std::lock_guard<std::mutex> lk(mutex);
    pair = eventPool.acquire();
    pendingEvents.emplace(rec_ptr->launch_id, pair);
    pendingStack[stream].push_back(PendingRecord{rec_ptr->launch_id, rec_ptr});
  }

  CUDA_SAFECALL(cudaEventRecord(pair.start, (cudaStream_t)stream));

  // Optional: you can fill start timestamp here, but it's more consistent to do
  // it in the callback (after both events exist). We'll leave it 0 for now.
  rec_ptr->gpuStartCycles = 0;
  rec_ptr->gpuEndCycles = 0;
}

static void on_kernel_exit(CUcontext /*ctx*/, CUstream stream,
                           const char* /*kernel_name*/, void** kernel_params,
                           CUresult* /*pResult*/) {
  stream = reinterpret_cast<cuLaunchKernel_params_st*>(*kernel_params)->hStream;
  PendingRecord pending{};
  CudaEventPair pair{};

  {
    std::lock_guard<std::mutex> lk(mutex);

    auto sit = pendingStack.find(stream);
    assert(sit != pendingStack.end());
    assert(!sit->second.empty());

    pending = sit->second.back();
    sit->second.pop_back();
    if (sit->second.empty()) {
      // keep map small
      pendingStack.erase(sit);
    }

    auto eit = pendingEvents.find(pending.launchId);
    assert(eit != pendingEvents.end());
    pair = eit->second;
    pendingEvents.erase(eit);
  }

  // Record end event *after* the kernel in the same stream.
  CUDA_SAFECALL(cudaEventRecord(pair.end, (cudaStream_t)stream));

  // Enqueue a host callback after everything prior in the stream is done.
  auto* payload = new KernelDonePayload{
      .events = pair,
      .rec = pending.rec,
  };
  CUDA_SAFECALL(
      cudaLaunchHostFunc((cudaStream_t)stream, kernel_done_callback, payload));
}

// -----------------------------
// Public entrypoint used by your NVBit handler layer
// -----------------------------
//
// IMPORTANT: This file doesn't know *exactly* how you're intercepting cuLaunchKernel.
// Typically you do something like:
//   - call on_kernel_enter(...) before the real cuLaunchKernel
//   - call the real cuLaunchKernel
//   - call on_kernel_exit(...) after
//
// Wire this function from your dispatcher accordingly.
void handleKernelLaunch(CUcontext ctx, int is_exit, const char* kernel_name,
                        void* kernel_params, CUresult* pResult) {
  // This assumes kernel_params is the address of the "void** kernelParams" pointer
  // that points to cuLaunchKernel_params_st. This matches your earlier code:
  //   cuLaunchKernel_params_st* params = (cuLaunchKernel_params_st*)(*kernel_params);
  //
  // If your actual interception passes something else, adjust here.
  void** kp = reinterpret_cast<void**>(&kernel_params);

  if (is_exit) {
    on_kernel_exit(ctx,
                   (CUstream) nullptr /* stream filled by your dispatcher? */,
                   kernel_name, kp, pResult);
    return;
  }
  // Enter hook
  on_kernel_enter(ctx,
                  (CUstream) nullptr /* stream filled by your dispatcher? */,
                  kernel_name, kp);

  // NOTE: We do NOT call the real launch here because that depends on your
  // NVBit interception mechanism / dispatch table.
  //
  // In your actual wrapper you likely have:
  //   REAL_cuLaunchKernel(..., pResult)
  //
  // Then you call:
  //   on_kernel_exit(...)

  (void)pResult;
}

void collectCompletedKernelLaunches(std::vector<KernelLaunchInfo>& out) {
  std::lock_guard<std::mutex> lk(mutex);
  out.insert(out.end(), completedKernelLaunches.begin(),
             completedKernelLaunches.end());
  completedKernelLaunches.clear();
}

void printCompletedKernelLaunches(FILE* out) {
  std::vector<KernelLaunchInfo> records;
  collectCompletedKernelLaunches(records);

  fprintf(out, "===== Completed Kernel Launches (%zu) =====\n", records.size());

  for (const auto& r : records) {
    uint64_t durNs = (r.gpuEndCycles > r.gpuStartCycles)
                         ? (r.gpuEndCycles - r.gpuStartCycles)
                         : 0;

    fprintf(out,
            "kernel=%s "
            "launchId=%lu "
            "stream=%p "
            "grid=(%u,%u,%u) "
            "block=(%u,%u,%u) "
            "shmem=%uB "
            "startNs=%lu "
            "endNs=%lu "
            "durNs=%lu\n",
            r.kernel_name ? r.kernel_name : "<unknown>", r.launch_id,
            (void*)r.stream, r.gridX, r.gridY, r.gridZ, r.blockX, r.blockY,
            r.blockZ, r.sharedMemBytes, r.gpuStartCycles, r.gpuEndCycles,
            durNs);
  }

  fflush(out);
}

}  // namespace notrace
