#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kv_emulation_kernel(int* data, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        data[idx] = (data[idx] * 3) + 7;
    }
}

#define CHECK_CUDA(cmd) { \
    cudaError_t error = cmd; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CU(cmd) { \
    CUresult result = cmd; \
    if (result != CUDA_SUCCESS) { \
        std::cerr << "Driver API Error Code: " << result << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    CHECK_CU(cuInit(0));
    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    std::cout << "====================================================================" << std::endl;
    std::cout << "EDMM End-to-End Silicon Benchmark" << std::endl;
    std::cout << "====================================================================" << std::endl;

    size_t page_size = 2 * 1024 * 1024;
    size_t total_size = 2 * page_size;
    int element_count = total_size / sizeof(int);
    int half_elements = element_count / 2;

    // Reserve contiguous VA range
    CUdeviceptr virtual_base;
    CHECK_CU(cuMemAddressReserve(&virtual_base, total_size, 0, 0, 0));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    // Three independent physical handles
    CUmemGenericAllocationHandle handle_invariant;
    CUmemGenericAllocationHandle handle_recompute_full;
    CUmemGenericAllocationHandle handle_speculative;

    CHECK_CU(cuMemCreate(&handle_invariant, page_size, &prop, 0));
    CHECK_CU(cuMemCreate(&handle_recompute_full, total_size, &prop, 0));
    CHECK_CU(cuMemCreate(&handle_speculative, page_size, &prop, 0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    float ms = 0.0f;

    int threads = 256;
    int blocks_full = (element_count + threads - 1) / threads;
    int blocks_half = (half_elements + threads - 1) / threads;

    const int NUM_TRIALS = 10;
    float trials_a[NUM_TRIALS], trials_b[NUM_TRIALS], trials_c[NUM_TRIALS];

    // Warmup: map, run kernel, unmap
    CHECK_CU(cuMemMap(virtual_base, page_size, 0, handle_invariant, 0));
    CHECK_CU(cuMemMap(virtual_base + page_size, page_size, 0, handle_speculative, 0));
    CHECK_CU(cuMemSetAccess(virtual_base, total_size, &access, 1));
    process_kv_emulation_kernel<<<blocks_full, threads>>>((int*)virtual_base, element_count);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CU(cuMemUnmap(virtual_base, total_size));

    std::cout << "\nRunning " << NUM_TRIALS << " trials per group...\n" << std::endl;

    // ------------------------------------------------------------------
    // Group A: Contiguous Cache Hit (invariant + speculative pre-mapped)
    // ------------------------------------------------------------------
    for (int t = 0; t < NUM_TRIALS; t++) {
        CHECK_CU(cuMemMap(virtual_base, page_size, 0, handle_invariant, 0));
        CHECK_CU(cuMemMap(virtual_base + page_size, page_size, 0, handle_speculative, 0));
        CHECK_CU(cuMemSetAccess(virtual_base, total_size, &access, 1));

        CHECK_CUDA(cudaEventRecord(t0));
        process_kv_emulation_kernel<<<blocks_full, threads>>>((int*)virtual_base, element_count);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        trials_a[t] = ms * 1000.0f;

        CHECK_CU(cuMemUnmap(virtual_base, total_size));
    }

    // ------------------------------------------------------------------
    // Group B: Full Recompute (memcpy fresh data + kernel over full range)
    // ------------------------------------------------------------------
    int* host_buf = new int[element_count]();
    for (int i = 0; i < element_count; i++) host_buf[i] = i;

    for (int t = 0; t < NUM_TRIALS; t++) {
        CHECK_CU(cuMemMap(virtual_base, total_size, 0, handle_recompute_full, 0));
        CHECK_CU(cuMemSetAccess(virtual_base, total_size, &access, 1));

        CHECK_CUDA(cudaEventRecord(t0));
        CHECK_CUDA(cudaMemcpy((void*)virtual_base, host_buf, total_size, cudaMemcpyHostToDevice));
        process_kv_emulation_kernel<<<blocks_full, threads>>>((int*)virtual_base, element_count);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        trials_b[t] = ms * 1000.0f;

        CHECK_CU(cuMemUnmap(virtual_base, total_size));
    }
    delete[] host_buf;

    // ------------------------------------------------------------------
    // Group C: EDMM Virtual Page Swap (remap second page, kernel on suffix only)
    // ------------------------------------------------------------------
    for (int t = 0; t < NUM_TRIALS; t++) {
        // Pre-map invariant page (done during orchestration bubble)
        CHECK_CU(cuMemMap(virtual_base, page_size, 0, handle_invariant, 0));
        CHECK_CU(cuMemSetAccess(virtual_base, page_size, &access, 1));

        // Simulate tool execution pause
        usleep(1000);

        CHECK_CUDA(cudaEventRecord(t0));
        // EDMM operation: map the speculative page into the second slot
        CHECK_CU(cuMemMap(virtual_base + page_size, page_size, 0, handle_speculative, 0));
        CHECK_CU(cuMemSetAccess(virtual_base + page_size, page_size, &access, 1));
        // Only compute the suffix portion
        process_kv_emulation_kernel<<<blocks_half, threads>>>((int*)(virtual_base + page_size), half_elements);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
        trials_c[t] = ms * 1000.0f;

        CHECK_CU(cuMemUnmap(virtual_base, total_size));
    }

    // ------------------------------------------------------------------
    // Results
    // ------------------------------------------------------------------
    auto mean = [](float* v, int n) { float s = 0; for (int i = 0; i < n; i++) s += v[i]; return s / n; };
    auto stddev = [&mean](float* v, int n) {
        float m = mean(v, n), s = 0;
        for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
        return sqrtf(s / n);
    };

    float mu_a = mean(trials_a, NUM_TRIALS), sd_a = stddev(trials_a, NUM_TRIALS);
    float mu_b = mean(trials_b, NUM_TRIALS), sd_b = stddev(trials_b, NUM_TRIALS);
    float mu_c = mean(trials_c, NUM_TRIALS), sd_c = stddev(trials_c, NUM_TRIALS);

    std::cout << "-------------------- HARDWARE PROFILING RESULTS --------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Group | Description                    | Mean (us) | StdDev (us) |" << std::endl;
    std::cout << "|-------|--------------------------------|-----------|-------------|" << std::endl;
    std::cout << "| A     | Contiguous Cache Hit           | " << std::setw(9) << mu_a << " | " << std::setw(11) << sd_a << " |" << std::endl;
    std::cout << "| B     | Full Recompute (memcpy+kernel) | " << std::setw(9) << mu_b << " | " << std::setw(11) << sd_b << " |" << std::endl;
    std::cout << "| C     | EDMM Page Swap (map+suffix)    | " << std::setw(9) << mu_c << " | " << std::setw(11) << sd_c << " |" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;
    std::cout << "\n  Radix Penalty (B/A): " << (mu_b / mu_a) << "x" << std::endl;
    std::cout << "  EDMM Recovery (C/A): " << (mu_c / mu_a) << "x" << std::endl;

    std::cout << "\n  Per-trial detail (us):" << std::endl;
    std::cout << "  Trial |    A    |    B    |    C    " << std::endl;
    std::cout << "  ------|---------|---------|--------" << std::endl;
    for (int t = 0; t < NUM_TRIALS; t++) {
        std::cout << "    " << std::setw(2) << t+1 << "  | "
                  << std::setw(7) << trials_a[t] << " | "
                  << std::setw(7) << trials_b[t] << " | "
                  << std::setw(7) << trials_c[t] << std::endl;
    }

    std::cout << "\n====================================================================" << std::endl;

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CU(cuMemAddressFree(virtual_base, total_size));
    CHECK_CU(cuMemRelease(handle_invariant));
    CHECK_CU(cuMemRelease(handle_recompute_full));
    CHECK_CU(cuMemRelease(handle_speculative));
    CHECK_CU(cuCtxDestroy(context));

    return 0;
}
