#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib> 
#include <ctime>
#include <iomanip>
#include <string>

const int BLOCK_SIZE = 16;
const int REPETITIONS = 10;
const int DIMS[] = { 100, 200, 400, 800, 1600, 2000 };

// Матричное произведение на хосте
void matmul_cpu(float* A, float* B, float* C, int I, int J, int K) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            float prod = 0.0f;
            for (int k = 0; k < K; k++) {
                prod += A[i*K + k] * B[k*J + j];
            }
            C[i*J + j] = prod;
        }
    }
}

// Матричное произведение на девайсе
__global__ void matmul_gpu(float* A, float* B, float* C, int I, int J, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < I && j < J) {
        float prod = 0.0f;
        for (int k = 0; k < K; k++) {
            prod += A[i*K + k] * B[k*J + j];
        }
        C[i*J + j] = prod;
    }
}

// Заполнение матрицы случайными значениями
void generate_mat(float* mat, int I, int J) {
    for (int i = 0; i < I*J; i++) {
        mat[i] = rand() / (float) RAND_MAX;
    }
}

// Проверка результата
bool check_result(float* src, float* dst, int I, int J, float eps=1e-3f) {
    for (int i = 0; i < I * J; i++) {
        if (fabs(src[i] - dst[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    cudaEvent_t gpu_time_start, gpu_time_stop;
    cudaEventCreate(&gpu_time_start);
    cudaEventCreate(&gpu_time_stop);

    std::cout << std::left
        << std::setw(10) << "DIM"
        << std::setw(10) << "CPU, ms"
        << std::setw(10) << "GPU, ms"
        << std::setw(9) << "S"
        << "Is correct" << std::endl;

    for (int dim : DIMS) {
        int I = dim, J = dim, K = dim;

        // Инициализации массивов хоста
        float* h_A = new float[I*K];
        float* h_B = new float[K*J];
        float* h_C_cpu = new float[I*J];
        float* h_C_gpu = new float[I*J];
        
        // Инициализация массивов девайса
        float* d_A, * d_B, * d_C;
        size_t size_A = I * K * sizeof(float);
        size_t size_B = K * J * sizeof(float);
        size_t size_C = I * J * sizeof(float);
        cudaMalloc(&d_A, size_A);
        cudaMalloc(&d_B, size_B);
        cudaMalloc(&d_C, size_C);

        // Определение размерностей блоков и сетки
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((I + BLOCK_SIZE - 1) / BLOCK_SIZE, (J + BLOCK_SIZE - 1) / BLOCK_SIZE);

        double cpu_total_time = 0.0;
        double gpu_total_time = 0.0;

        bool isCorrect = true;

        for (int rep = 0; rep < REPETITIONS; rep++) {
            // Заполнение матриц случайными значениями
            generate_mat(h_A, I, K);
            generate_mat(h_B, K, J);

            // Вычисление на хосте
            auto cpu_time_start = std::chrono::high_resolution_clock::now();
            matmul_cpu(h_A, h_B, h_C_cpu, I, J, K);
            auto cpu_time_end = std::chrono::high_resolution_clock::now();
            cpu_total_time += std::chrono::duration<double, std::milli>(cpu_time_end - cpu_time_start).count();

            // Копирование массивов с хоста на девайс
            cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

            // Вычисление на девайсе
            cudaEventRecord(gpu_time_start);
            matmul_gpu<<<grid, block>>>(d_A, d_B, d_C, I, J, K);
            cudaEventRecord(gpu_time_stop);
            cudaDeviceSynchronize();

            float gpu_time;
            cudaEventElapsedTime(&gpu_time, gpu_time_start, gpu_time_stop);
            gpu_total_time += gpu_time;

            // Копирование результата с девайса на хост
            cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

            // Проверка результата, полученного с девайса
            if (!check_result(h_C_cpu, h_C_gpu, I, J)) {
                isCorrect = false;
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        delete[] h_A;
        delete[] h_B;
        delete[] h_C_cpu;
        delete[] h_C_gpu;

        // Подсчет среднего времени вычислений на хосте и девайсе, а также ускорения
        double cpu_avg_time = cpu_total_time / REPETITIONS;
        double gpu_avg_time = gpu_total_time / REPETITIONS;
        double S = cpu_avg_time / gpu_avg_time;

        std::cout << std::fixed << std::setprecision(2) << std::left 
            << std::setw(10) << (std::to_string(dim) + "x" + std::to_string(dim))
            << std::setw(10) << cpu_avg_time
            << std::setw(10) << gpu_avg_time
            << std::setw(9) << S
            << "  " << (isCorrect ? "YES" : "NO") << std::endl;
    }

    cudaEventDestroy(gpu_time_start);
    cudaEventDestroy(gpu_time_stop);
}