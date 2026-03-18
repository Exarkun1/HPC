#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iomanip>
#include <string>

const size_t BLOCK_SIZE = 512;
const size_t GRID_SIZE = 1024;
const size_t REPETITIONS = 10;

double pi_calc_cpu(unsigned long long N) {
    unsigned long long hits = 0;

    for (unsigned long long i = 0; i < N; i++) {
        double x = (double)rand() / (double)RAND_MAX;
        double y = (double)rand() / (double)RAND_MAX;
        if (x*x + y*y <= 1.0f) {
            ++hits;
        }
    }

    return 4.0 * (double)hits / (double)N;
}

__global__ void pi_calc_kernel(unsigned long long N, unsigned long long* hits, unsigned long long seed) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, i, 0, &state);

    unsigned long long local_hits = 0;
    for (unsigned long long i = 0; i < N; i++) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x*x + y*y <= 1.0f) {
            ++local_hits;
        }
    }

    extern __shared__ unsigned long long shared_hits[];
    shared_hits[threadIdx.x] = local_hits;
    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j /= 2) {
        if (threadIdx.x < j) {
            shared_hits[threadIdx.x] += shared_hits[threadIdx.x + j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hits[blockIdx.x] = shared_hits[0];
    }
}

double pi_calc_gpu(unsigned long long N) {
    unsigned long long local_N = N / (GRID_SIZE * BLOCK_SIZE);
    size_t local_mem_size = BLOCK_SIZE * sizeof(unsigned long long);

    unsigned long long *d_hits;
    size_t hits_size = GRID_SIZE * sizeof(unsigned long long);

    unsigned long long* h_hits = new unsigned long long[GRID_SIZE];
    cudaMalloc(&d_hits, hits_size);

    pi_calc_kernel<<<GRID_SIZE, BLOCK_SIZE, local_mem_size>>>(local_N, d_hits, 1234);
    cudaDeviceSynchronize();

    cudaMemcpy(h_hits, d_hits, hits_size, cudaMemcpyDeviceToHost);
    
    unsigned long long total_hits = 0;
    for (unsigned long long i = 0; i < GRID_SIZE; i++) {
        total_hits += h_hits[i];
    }

    double pi_est = 4.0 * total_hits / N;

    cudaFree(d_hits);
    delete[] h_hits;

    return pi_est;
}

int main() {
    unsigned long long N = 1073741824ULL;
    srand(1234);
    double pi_cpu = pi_calc_cpu(N);
    std::cout << "CPU: " << pi_cpu << std::endl;
    double pi_gpu = pi_calc_gpu(N);
    std::cout << "GPU: " << pi_gpu << std::endl;
    std::cout << "Difference: " << std::abs(pi_gpu - pi_cpu) << std::endl;
    return 0;
}