#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <chrono>
#include <iomanip>

const size_t BLOCK_SIZE = 512;
const size_t GRID_SIZE = 1024;
const size_t REPETITIONS = 10;
const unsigned long long N_POINTS[] = { 524288, 2097152, 8388608, 33554432, 134217728, 536870912 };

// Вычисление числа pi методом Монте-Карло на хосте
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

// Ядро вычисление числа pi методом Монте-Карло на девайсе
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

// Вычисление числа pi методом Монте-Карло на девайсе
double pi_calc_gpu(unsigned long long N, double* kernel_time=nullptr) {
    unsigned long long local_N = N / (GRID_SIZE * BLOCK_SIZE);
    size_t local_mem_size = BLOCK_SIZE * sizeof(unsigned long long);

    unsigned long long *d_hits;
    size_t hits_size = GRID_SIZE * sizeof(unsigned long long);

    unsigned long long* h_hits = new unsigned long long[GRID_SIZE];
    cudaMalloc(&d_hits, hits_size);

    auto time_start = std::chrono::high_resolution_clock::now();
    pi_calc_kernel<<<GRID_SIZE, BLOCK_SIZE, local_mem_size>>>(local_N, d_hits, 1234);
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();
    if (kernel_time != nullptr) {
        *kernel_time += std::chrono::duration<double, std::milli>(time_end - time_start).count();
    }

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
    std::cout << std::left
        << std::setw(15) << "N"
        << std::setw(10) << "CPU, ms"
        << std::setw(10) << "GPU, ms"
        << std::setw(15) << "GPU ker, ms"
        << std::setw(10) << "S"
        << std::setw(10) << "S ker"
        << std::setw(15) << "CPU pi"
        << std::setw(15) << "GPU pi" << std::endl;

    for (unsigned long long N : N_POINTS) {
        double cpu_total_time = 0.0;
        double gpu_total_time = 0.0;
        double gpu_kernel_total_time = 0.0;

        double pi_est_cpu = 0.0f;
        double pi_est_gpu = 0.0f;

        for (unsigned int rep = 0; rep < REPETITIONS; rep++) {
            // Вычисление на хосте
            auto cpu_time_start = std::chrono::high_resolution_clock::now();
            pi_est_cpu = pi_calc_cpu(N);
            auto cpu_time_end = std::chrono::high_resolution_clock::now();
            cpu_total_time += std::chrono::duration<double, std::milli>(cpu_time_end - cpu_time_start).count();

            // Вычисление на девайсе
            auto gpu_time_start = std::chrono::high_resolution_clock::now();
            pi_est_gpu = pi_calc_gpu(N, &gpu_kernel_total_time);
            auto gpu_time_end = std::chrono::high_resolution_clock::now();
            gpu_total_time += std::chrono::duration<double, std::milli>(gpu_time_end - gpu_time_start).count();
        }

        // Подсчет среднего времени вычислений на хосте и девайсе, а также ускорения
        double cpu_avg_time = cpu_total_time / REPETITIONS;
        double gpu_avg_time = gpu_total_time / REPETITIONS;
        double gpu_kernel_avg_time = gpu_kernel_total_time / REPETITIONS;
        double S = cpu_avg_time / gpu_avg_time;
        double Sk = cpu_avg_time / gpu_kernel_avg_time;

        std::cout << std::fixed << std::setprecision(2) << std::left 
            << std::setw(15) << N
            << std::setw(10) << cpu_avg_time
            << std::setw(10) << gpu_avg_time
            << std::setw(15) << gpu_kernel_avg_time
            << std::setw(10) << S
            << std::setw(10) << Sk
            << std::setprecision(7)
            << std::setw(15) << pi_est_cpu
            << std::setw(15) << pi_est_gpu << std::endl;
    }
}