#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "cuda_event_handlers.h"
#include "event.h"
#include "nvbit.h"
#include "nvbit_tool.h"
#include "utils/mpsc_queue.h"  // Ensure this includes your Queue definition

static std::atomic<bool> g_stop_logging{false};
static std::thread* g_logger_thread = nullptr;

void logging_worker_loop() {
  auto& queue = notrace::MPSCMessageQueue::getInstance();

  while (!g_stop_logging.load(std::memory_order_relaxed)) {
    size_t bytes_processed = queue.processUpdates();

    if (bytes_processed == 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  queue.processUpdates();
}

void nvbit_at_init() {
  printf("Tool is being loaded\n");

  notrace::register_handlers();

  notrace::MPSCMessageQueue::getInstance().registerConsumers();

  g_logger_thread = new std::thread(logging_worker_loop);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {

  if constexpr (notrace::debug::ENABLE_DEBUG_LOGS)
    printf("nvbit_at_cuda_event: %s %s\n", is_exit ? "EXIT" : "ENTER", name);
  notrace::cuda_event_handler_t* handler =
      notrace::get_cuda_event_handler(cbid);

  // If we have a custom handler (like KernelLaunch), run it.
  if (handler != nullptr) {
    return handler(ctx, is_exit, name, params, pStatus);
  }
}

void nvbit_at_term() {
  // 1. Synchronize the device to ensure all kernels finished
  cudaDeviceSynchronize();

  printf("Tool is being unloaded. Stopping logger...\n");

  // 2. Signal the logger thread to stop
  g_stop_logging.store(true);

  // 3. Join the thread (wait for it to finish flushing)
  if (g_logger_thread) {
    if (g_logger_thread->joinable()) {
      g_logger_thread->join();
    }
    delete g_logger_thread;
    g_logger_thread = nullptr;
  }

  printf("Logger stopped. Output flushed.\n");
  fflush(stdout);
}

// Fake function to force nvbit is linked and not optimized out
void __fake_nvbit_loader(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);
}
