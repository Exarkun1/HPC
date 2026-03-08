#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <string>

const size_t BLOCK_SIZE = 512;
const size_t REPETITIONS = 10;
const size_t SIZES[] = { 1000, 5000, 10000, 50000, 100000, 500000, 1000000 };

// Сложение элементов вектора на хосте
float vector_sum_cpu(const float* a, unsigned int I) {
    float sum_a = 0.0f;
    for (unsigned int i = 0; i < I; i++) {
        sum_a += a[i];
    }
    return sum_a;
} 

// Ядро сложения элементов вектора на девайсе
__global__ void vector_sum_kernel(const float* a, float* sum_a, unsigned int I) {
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
        atomicAdd(sum_a, local_sums[0]);
    }
}

// Сложение элементов вектора на девайсе
float vector_sum_gpu(float* h_a, unsigned int I, double* kernel_time=nullptr) {
    size_t grid_size = (I + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t local_mem_size = BLOCK_SIZE * sizeof(float);

    float* d_a, * d_sum_a;
    size_t size_a = I * sizeof(float);
    size_t size_sum_a = 1 * sizeof(float);
    
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_sum_a, size_sum_a);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);

    auto time_start = std::chrono::high_resolution_clock::now();
    vector_sum_kernel<<<grid_size, BLOCK_SIZE, local_mem_size>>>(d_a, d_sum_a, I);
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();
    if (kernel_time != nullptr) {
        *kernel_time += std::chrono::duration<double, std::milli>(time_end - time_start).count();
    }

    float h_sum_a = 0.0f;
    cudaMemcpy(&h_sum_a, d_sum_a, size_sum_a, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_sum_a);

    return h_sum_a;
}

// Заполнения вектора единицами
void fill_vector_ones(float* vector, unsigned int I) {
    for (unsigned int i = 0; i < I; i++) {
        vector[i] = 1.0f;
    }
}

int main() {
    std::cout << std::left
        << std::setw(10) << "SIZE"
        << std::setw(10) << "CPU, ms"
        << std::setw(10) << "GPU, ms"
        << std::setw(15) << "GPU ker, ms"
        << std::setw(10) << "S"
        << std::setw(10) << "S ker"
        << std::setw(15) << "CPU sum"
        << std::setw(15) << "GPU sum" << std::endl;

    for (size_t size : SIZES) {
        // Инициализации массива хоста
        float* h_a = new float[size];
        fill_vector_ones(h_a, size);

        double cpu_total_time = 0.0;
        double gpu_total_time = 0.0;
        double gpu_kernel_total_time = 0.0;

        float sum_a_cpu = 0.0f;
        float sum_a_gpu = 0.0f;

        for (unsigned int rep = 0; rep < REPETITIONS; rep++) {
            // Вычисление на хосте
            auto cpu_time_start = std::chrono::high_resolution_clock::now();
            sum_a_cpu = vector_sum_cpu(h_a, size);
            auto cpu_time_end = std::chrono::high_resolution_clock::now();
            cpu_total_time += std::chrono::duration<double, std::milli>(cpu_time_end - cpu_time_start).count();

            // Вычисление на девайсе
            auto gpu_time_start = std::chrono::high_resolution_clock::now();
            sum_a_gpu = vector_sum_gpu(h_a, size, &gpu_kernel_total_time);
            auto gpu_time_end = std::chrono::high_resolution_clock::now();
            gpu_total_time += std::chrono::duration<double, std::milli>(gpu_time_end - gpu_time_start).count();
        }

        delete[] h_a;

        // Подсчет среднего времени вычислений на хосте и девайсе, а также ускорения
        double cpu_avg_time = cpu_total_time / REPETITIONS;
        double gpu_avg_time = gpu_total_time / REPETITIONS;
        double gpu_kernel_avg_time = gpu_kernel_total_time / REPETITIONS;
        double S = cpu_avg_time / gpu_avg_time;
        double Sk = cpu_avg_time / gpu_kernel_avg_time;

        std::cout << std::fixed << std::setprecision(2) << std::left 
            << std::setw(10) << std::to_string(size)
            << std::setw(10) << cpu_avg_time
            << std::setw(10) << gpu_avg_time
            << std::setw(15) << gpu_kernel_avg_time
            << std::setw(10) << S
            << std::setw(10) << Sk
            << std::setw(15) << sum_a_cpu
            << std::setw(15) << sum_a_gpu << std::endl;
    }
    
}