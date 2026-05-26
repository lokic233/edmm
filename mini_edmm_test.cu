#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void verify_edmm_kernel(int* data, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = data[idx] * 2;
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
    std::cout << "Nano-EDMM Bare-Metal Page Aliasing Smoke Test" << std::endl;
    std::cout << "====================================================================" << std::endl;

    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;

    CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    std::cout << "[VMM] Device allocation granularity: " << granularity / 1024 << " KB" << std::endl;

    size_t page_size = (granularity > 2 * 1024 * 1024) ? granularity : 2 * 1024 * 1024;
    size_t total_size = 2 * page_size;
    int element_count = total_size / sizeof(int);

    // Step 1: Reserve contiguous virtual address space
    CUdeviceptr virtual_ptr;
    CHECK_CU(cuMemAddressReserve(&virtual_ptr, total_size, 0, 0, 0));
    std::cout << "[VMM] Reserved VA range: 0x" << std::hex << virtual_ptr
              << " - 0x" << (virtual_ptr + total_size) << std::dec
              << " (" << total_size / (1024*1024) << " MB)" << std::endl;

    // Step 2: Allocate two independent physical handles
    CUmemGenericAllocationHandle handle_A, handle_B;
    CHECK_CU(cuMemCreate(&handle_A, page_size, &prop, 0));
    CHECK_CU(cuMemCreate(&handle_B, page_size, &prop, 0));
    std::cout << "[PHYSICAL] Handle A: " << page_size / (1024*1024) << " MB (invariant KV cache block)" << std::endl;
    std::cout << "[PHYSICAL] Handle B: " << page_size / (1024*1024) << " MB (dynamic tool response block)" << std::endl;

    // Step 3: Map physical handles into the contiguous VA range
    CHECK_CU(cuMemMap(virtual_ptr, page_size, 0, handle_A, 0));
    CHECK_CU(cuMemMap(virtual_ptr + page_size, page_size, 0, handle_B, 0));
    std::cout << "[MAPPING] Two physical allocations mapped into one contiguous VA pointer" << std::endl;

    // Step 4: Set access permissions
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = 0;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(virtual_ptr, total_size, &accessDesc, 1));

    // Step 5: Fill from host
    int* host_buffer = new int[element_count];
    for (int i = 0; i < element_count; i++) {
        host_buffer[i] = i;
    }
    CHECK_CUDA(cudaMemcpy((void*)virtual_ptr, host_buffer, total_size, cudaMemcpyHostToDevice));

    // Step 6: Kernel launch across the physical page boundary
    int threads = 256;
    int blocks = (element_count + threads - 1) / threads;
    std::cout << "[KERNEL] Launching " << blocks << " blocks x " << threads << " threads ("
              << element_count << " elements)" << std::endl;
    verify_edmm_kernel<<<blocks, threads>>>((int*)virtual_ptr, element_count);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 7: Verify results
    CHECK_CUDA(cudaMemcpy(host_buffer, (void*)virtual_ptr, total_size, cudaMemcpyDeviceToHost));

    int boundary = element_count / 2;
    std::cout << "\n[VERIFICATION]:" << std::endl;
    std::cout << "  Element 0          (Handle A start): " << host_buffer[0]
              << "  (expected 0)" << std::endl;
    std::cout << "  Element " << boundary - 1 << "  (Handle A end):   " << host_buffer[boundary - 1]
              << "  (expected " << (boundary - 1) * 2 << ")" << std::endl;
    std::cout << "  Element " << boundary << "  (Handle B start): " << host_buffer[boundary]
              << "  (expected " << boundary * 2 << ")" << std::endl;
    std::cout << "  Element " << element_count - 1 << " (Handle B end):   " << host_buffer[element_count - 1]
              << "  (expected " << (element_count - 1) * 2 << ")" << std::endl;

    bool pass = true;
    for (int i = 0; i < element_count; i++) {
        if (host_buffer[i] != i * 2) {
            std::cerr << "  MISMATCH at index " << i << ": got " << host_buffer[i]
                      << ", expected " << i * 2 << std::endl;
            pass = false;
            break;
        }
    }

    if (pass) {
        std::cout << "\n[PASS] Kernel executed across physical page boundary with zero faults." << std::endl;
        std::cout << "       VMM page aliasing verified: two independent physical allocations" << std::endl;
        std::cout << "       appear as one contiguous buffer to the GPU compute grid." << std::endl;
    } else {
        std::cout << "\n[FAIL] Data mismatch detected." << std::endl;
    }

    // Cleanup
    delete[] host_buffer;
    CHECK_CU(cuMemUnmap(virtual_ptr, total_size));
    CHECK_CU(cuMemAddressFree(virtual_ptr, total_size));
    CHECK_CU(cuMemRelease(handle_A));
    CHECK_CU(cuMemRelease(handle_B));
    CHECK_CU(cuCtxDestroy(context));

    return pass ? 0 : 1;
}
