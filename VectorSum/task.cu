#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const size_t BLOCK_SIZE = 512;
const size_t REPETITIONS = 10;
const size_t SIZES[] = { 1000, 5000, 10000, 50000, 100000, 500000, 1000000 };

float vector_sum_cpu(const float* a, int I) {
    float sum_a = 0.0f;
    for (unsigned int i = 0; i < I; i++) {
        sum_a += a[i];
    }
    return sum_a;
} 

__global__ void vector_sum_kernel(const float* a, float* partial_sums, int I) {
    extern __shared__ float local_sums[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    local_sums[threadIdx.x] = (i < I) ? a[i] : 0.0f;
    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j /= 2) {
        if (threadIdx.x < j) {
            local_sums[threadIdx.x] += local_sums[threadIdx.x + j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = local_sums[0];
    }
}

float vector_sum_gpu(float* h_a, int I) {
    size_t grid_size = (I + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t local_mem_size = BLOCK_SIZE * sizeof(float);

    float* d_a, * d_partial_sums;
    size_t size_a = I * sizeof(float);
    size_t size_partial_sums = grid_size * sizeof(float);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_partial_sums, size_partial_sums);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);

    vector_sum_kernel<<<grid_size, BLOCK_SIZE, local_mem_size>>>(d_a, d_partial_sums, I);
    cudaDeviceSynchronize();

    float* h_partial_sums = new float[grid_size];
    cudaMemcpy(h_partial_sums, d_partial_sums, size_partial_sums, cudaMemcpyDeviceToHost);

    float sum_a = 0.0f;
    for (unsigned int i = 0; i < grid_size; i++) {
        sum_a += h_partial_sums[i];
    }

    cudaFree(d_a);
    cudaFree(d_partial_sums);

    delete[] h_partial_sums;

    return sum_a;
}

void fill_vector_ones(float* vector, int I) {
    for (int i = 0; i < I; i++) {
        vector[i] = 1.0f;
    }
}

int main() {
    int size = SIZES[6];
    float* h_a = new float[size];
    fill_vector_ones(h_a, size);
    float sum_a_cpu = vector_sum_cpu(h_a, size);
    float sum_a_gpu = vector_sum_gpu(h_a, size);
    std::cout << sum_a_cpu << std::endl; 
    std::cout << sum_a_gpu << std::endl; 
}