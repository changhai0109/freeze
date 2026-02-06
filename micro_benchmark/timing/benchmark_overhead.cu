#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iomanip>
#include <thread>
// #define NVPROF // Comment this line to enable CUPTI profiling guidance

// --- Configuration ---
const int N = 1024; // Matrix size (4096 x 4096)
const int NUM_RUNS = 1000;
const int WARMUP_RUNS = 100;

// Error checking macro
#define CUDA_CHECK(call)                                                                                    \
    do                                                                                                      \
    {                                                                                                       \
        cudaError_t err = call;                                                                             \
        if (err != cudaSuccess)                                                                             \
        {                                                                                                   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1);                                                                                        \
        }                                                                                                   \
    } while (0)

void warnup(cublasHandle_t handle, float *d_A, float *d_B, float *d_C, float alpha, float beta)
{
    for (int i = 0; i < WARMUP_RUNS; ++i)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper to calculate Mean and Std Dev
void printStats(const std::string &name, std::vector<double> &times, double baselineMean = -1.0)
{
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

    std::cout << std::left << std::setw(25) << name
              << std::setw(15) << mean
              << std::setw(15) << stdev;

    if (baselineMean > 0)
    {
        double overhead = ((mean - baselineMean) / baselineMean) * 100.0;
        std::cout << "+" << overhead << "%";
    }
    else
    {
        std::cout << "--";
    }
    std::cout << std::endl;
}

int main()
{
    // 1. Setup Resources
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    warnup(handle, d_A, d_B, d_C, alpha, beta);

    std::cout << "Running Benchmark (Matrix Size: " << N << "x" << N << ", Runs: " << NUM_RUNS << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Method"
              << std::setw(15) << "Avg Latency(ms)"
              << std::setw(15) << "Std Dev(ms)"
              << "Overhead" << std::endl;
    std::cout << "--------------------------------------------------------------------------------\n";

    // ==========================================================
    // 1. Baseline (No Timing inside loop)
    // ==========================================================
    // We measure the total wall-clock time of the loop and divide by N.
    // This removes per-operation CPU measurement overhead.
    warnup(handle, d_A, d_B, d_C, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous work is done before starting timing
    // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to stabilize timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all work is done
    auto end = std::chrono::high_resolution_clock::now();
    auto baseline_elapsed = end - start;
    double baseline_total_ms = std::chrono::duration<double, std::milli>(baseline_elapsed).count();
    double baseline_mean = baseline_total_ms / NUM_RUNS;
    printf("Baseline (Wall Clock): %.6f ms (Avg)\n", baseline_mean);

#ifndef NVPROF
    // ==========================================================
    // 2. CUDA Event (Synchronous)
    // ==========================================================
    // This mimics tools that block the CPU thread to read the timer immediately.

    warnup(handle, d_A, d_B, d_C, alpha, beta);
    std::vector<double> sync_times(NUM_RUNS);
    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous work is done before starting timing
    // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to stabilize timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);

        CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous work is done before starting timing
        cudaEventRecord(start_evt);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop_evt);
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the event to complete (synchronous read)

        float ms;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        sync_times[i] = ms;

        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all work is done
    end = std::chrono::high_resolution_clock::now();
    auto synchronized_elapsed = end - start;
    auto synchronized_total_ms = std::chrono::duration<double, std::milli>(synchronized_elapsed).count();
    auto synchronized_overhead = (synchronized_total_ms - baseline_total_ms) / baseline_total_ms * 100.0;
    auto synchronized_average_lat = std::accumulate(sync_times.begin(), sync_times.end(), 0.0) / NUM_RUNS;
    auto synchronized_stdev = std::sqrt(std::inner_product(sync_times.begin(), sync_times.end(), sync_times.begin(), 0.0) / NUM_RUNS - synchronized_average_lat * synchronized_average_lat);
    printf("CUDA Event (Synchronous): %.6f ms (Avg), Overhead: %.2f%%, standard deviation: %.6f ms\n", synchronized_average_lat, synchronized_overhead, synchronized_stdev);

    // ==========================================================
    // 3. CUDA Event (Async)
    // ==========================================================
    // This mimics optimized profiling: Record now, read later.
    std::vector<double> async_times(NUM_RUNS);

    std::vector<cudaEvent_t> start_events(NUM_RUNS);
    std::vector<cudaEvent_t> stop_events(NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&stop_events[i]);
    }
    warnup(handle, d_A, d_B, d_C, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous work is done before starting timing
    // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to stabilize timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cudaEventRecord(start_events[i]);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop_events[i]);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all events to complete
    end = std::chrono::high_resolution_clock::now();
    auto async_elapsed = end - start;
    auto async_total_ms = std::chrono::duration<double, std::milli>(async_elapsed).count();
    auto async_overhead = (async_total_ms - baseline_total_ms) / baseline_total_ms * 100.0;
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        float ms;
        cudaEventElapsedTime(&ms, start_events[i], stop_events[i]);
        async_times[i] = ms;
    }
    auto async_average_lat = std::accumulate(async_times.begin(), async_times.end(), 0.0) / NUM_RUNS;
    auto async_stdev = std::sqrt(std::inner_product(async_times.begin(), async_times.end(), async_times.begin(), 0.0) / NUM_RUNS - async_average_lat * async_average_lat);
    printf("CUDA Event (Async): %.6f ms (Avg), Overhead: %.2f%%, standard deviation: %.6f ms\n", async_average_lat, async_overhead, async_stdev);

    // ==========================================================
    // 4. CUPTI (Guidance)
    // ==========================================================
    // std::cout << "\n[Note for CUPTI]:\n"
    //           << "To measure CUPTI overhead, run the 'Baseline' binary using:\n"
    //           << "  nvprof ./benchmark  (for legacy CUPTI)\n"
    //           << "  nsys profile ./benchmark (for modern Nsight Systems)\n"
    //           << "Then compare the execution time reported there vs the Baseline printed above.\n";
#endif

    // ==========================================================
    // 1. Baseline (No Timing inside loop)
    // ==========================================================
    // We measure the total wall-clock time of the loop and divide by N.
    // This removes per-operation CPU measurement overhead.
    warnup(handle, d_A, d_B, d_C, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous work is done before starting timing
    // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay to stabilize timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all work is done
    end = std::chrono::high_resolution_clock::now();
    baseline_elapsed = end - start;
    baseline_total_ms = std::chrono::duration<double, std::milli>(baseline_elapsed).count();
    baseline_mean = baseline_total_ms / NUM_RUNS;
    printf("Baseline (Wall Clock): %.6f ms (Avg)\n", baseline_mean);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
