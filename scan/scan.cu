#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdio.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256

// debugging
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Copy the input array into the result array.
// We do this because the scan will modify result in-place.
__global__ void copyArrayKernel(const int* input, int* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = input[idx];
    }
}

// Set all entries in data[start ... end-1] to 0.
// This is used to zero out the padded part of the array
// when N is not already a power of 2.
__global__ void zeroPaddingKernel(int* data, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        data[idx] = 0;
    }
}

// One step of the upsweep phase.
// Each thread handles one useful operation.
__global__ void upsweepKernel(int* data, int offset, int rounded_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    long long step = (long long)2 * offset;
    long long i = (long long)1 * tid * step;

    if (i + step - 1 < rounded_length) {
        data[i + step - 1] += data[i + offset - 1];
    }
}

// Before downsweep, the last element must be set to 0
// to turn the tree sum into an exclusive scan.
__global__ void setLastElementZeroKernel(int* data, int rounded_length) {
    data[rounded_length - 1] = 0;
}

// One step of the downsweep phase.
__global__ void downsweepKernel(int* data, int offset, int rounded_length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    long long step = (long long)2 * offset;
    long long i = (long long)1 * tid * step;

    if (i + step - 1 < rounded_length) {
        int temp = data[i + offset - 1];
        data[i + offset - 1] = data[i + step - 1];
        data[i + step - 1] += temp;
    }
}

// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result) {
    // ECE476 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // The wrapper allocates the device arrays with length rounded up
    // to the next power of 2, so we can safely scan up to that size.
    int rounded_length = nextPow2(N);

    // Step 1: copy input into result
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    copyArrayKernel<<<blocks, THREADS_PER_BLOCK>>>(input, result, N);
    cudaCheckError(cudaGetLastError());


    // If the array was padded up to a power of 2, set the padded
    // entries to 0 so they do not affect the scan result.
    if (rounded_length > N) {
        int padding = rounded_length - N;
        int paddingBlocks = (padding + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        zeroPaddingKernel<<<paddingBlocks, THREADS_PER_BLOCK>>>(result, N, rounded_length);
        cudaCheckError(cudaGetLastError());

    }

    // Step 2: upsweep phase
    // Build partial sums up a tree.
    for (int offset = 1; offset <= rounded_length / 2; offset *= 2) {
        int numOperations = rounded_length / (2 * offset);
        int opBlocks = (numOperations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        upsweepKernel<<<opBlocks, THREADS_PER_BLOCK>>>(result, offset, rounded_length);
        cudaCheckError(cudaGetLastError());

    }

    // Step 3: set the last element to 0
    // note: makes final result EXCLUSIVE scan.
    setLastElementZeroKernel<<<1, 1>>>(result, rounded_length);
    cudaCheckError(cudaGetLastError());


    // Step 4: downsweep phase
    // Push prefix sums back down the tree.
    for (int offset = rounded_length / 2; offset >= 1; offset /= 2) {
        int numOperations = rounded_length / (2 * offset);
        int opBlocks = (numOperations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        downsweepKernel<<<opBlocks, THREADS_PER_BLOCK>>>(result, offset, rounded_length);
        cudaCheckError(cudaGetLastError());

    }

    cudaCheckError(cudaDeviceSynchronize());

}

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray) {
    int* device_result;
    int* device_input;
    int N = end - inarray;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void**)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void**)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}


// Build a flag array where:
// flags[i] = 1 if input[i] == input[i+1]
// flags[i] = 0 otherwise
__global__ void buildFlagsKernel(const int* input, int* flags, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length - 1) {
        flags[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
    } else if (idx == length - 1) {
        // Last position can never start a repeated pair
        flags[idx] = 0;
    }
}

// If flags[i] == 1, put i into output[scan(flags)[i]]
__global__ void scatterKernel(const int* flags, const int* scanned_flags,
                              int* output,
                              int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length && flags[idx] == 1) {
        output[scanned_flags[idx]] = idx;
    }
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    int blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    // stores 0 or 1 depending on whether input[i] == input[i+1]
    int* device_flags = nullptr;
    // exclusive scan of flags
    int* device_scanned_flags = nullptr;

    cudaMalloc((void**)&device_flags, length * sizeof(int));
    cudaMalloc((void**)&device_scanned_flags, length * sizeof(int));


    // build the flags array
    buildFlagsKernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, device_flags, length);
    cudaCheckError(cudaGetLastError());

    // exlcusive scan the flags
    thrust::device_ptr<int> flags_ptr(device_flags);
    thrust::device_ptr<int> scanned_ptr(device_scanned_flags);
    thrust::exclusive_scan(flags_ptr, flags_ptr + length, scanned_ptr);

    // scatter matching indices
    // If flags[i] = 1, then scanned_flags[i] tells us where index i
    // should go in the output array.
    scatterKernel<<<blocks, THREADS_PER_BLOCK>>>(
        device_flags,
        device_scanned_flags,
        device_output,
        length
    );
    cudaCheckError(cudaGetLastError());

    // Step 4: compute total number of repeats
    // The total count is: scanned_flags[last] + flags[last]
    int lastPossibleFlag = 0;
    int lastPossibleScan = 0;

    if (length >= 2) {
        cudaMemcpy(&lastPossibleFlag,
                   device_flags + (length - 2),
                   sizeof(int),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(&lastPossibleScan,
                   device_scanned_flags + (length - 2),
                   sizeof(int),
                   cudaMemcpyDeviceToHost);
    }

    cudaFree(device_flags);
    cudaFree(device_scanned_flags);

    return lastPossibleScan + lastPossibleFlag;
}

//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int* input, int length, int* output, int* output_length) {
    int* device_input;
    int* device_output;
    int rounded_length = nextPow2(length);

    cudaMalloc((void**)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void**)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
