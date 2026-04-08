#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <cuda_runtime.h>

const unsigned int ALPHABET_SIZE = 256;
const unsigned int BLOCK_SIZE = 256;

struct Pair {
    unsigned int n;
    unsigned int k;
};

unsigned int* build_Hs(unsigned int start, unsigned int end, unsigned int num) {
    unsigned int* Hs = new unsigned int[num];
    unsigned int step = (end - start) / (num - 1);
    for (unsigned int i = 0; i < num; i++) {
        Hs[i] = start + i * step;
    }
    return Hs;
}

void init_random_text(std::string& text, unsigned int H) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, ALPHABET_SIZE-1);

    for (unsigned int i = 0; i < H; i++) {
        text[i] = static_cast<char>(dist(gen));
    }
}

void init_random_patterns(std::vector<std::string>& patterns, unsigned int N, unsigned int min_len, unsigned int max_len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min_len, max_len);

    for (unsigned int i = 0; i < N; i++) {
        int len = dist(gen);
        std::string pattern(len, '\0');
        init_random_text(pattern, len);
        patterns.push_back(pattern);
    }
}

unsigned int* build_lens(const std::vector<std::string>& patterns) {
    unsigned int* lens = new unsigned int[patterns.size()];
    for (unsigned int i = 0; i < patterns.size(); i++) {
        lens[i] = patterns[i].size();
    }
    return lens;
}

void init_table(std::vector<Pair>* table, const std::vector<std::string>& patterns) {
    for (unsigned int i = 0; i < patterns.size(); i++) {
        for (unsigned int k = 0; k < patterns[i].size(); k++) {
            unsigned char c = patterns[i][k];
            table[c].push_back({i, k});
        }
    }
}

void init_R(int* R, const unsigned int* lens, unsigned int N, unsigned int H) {
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < H; j++) {
            R[i * H + j] = lens[i];
        }
    }
}

int init_pairs(std::vector<Pair>& pairs, unsigned int* offsets, unsigned int* sizes, const std::vector<Pair>* table) {
    unsigned int offset = 0;
    unsigned int max_size = 0;

    for (unsigned int c = 0; c < ALPHABET_SIZE; c++) {
        offsets[c] = offset;
        sizes[c] = table[c].size();
        max_size = max(max_size, sizes[c]);

        for (auto& pair : table[c]) {
            pairs.push_back(pair);
        }
        offset += sizes[c];
    }
    return max_size;
}

void mass_search_cpu(
    int* R,
    const char* text, 
    unsigned int H, 
    const Pair* pairs, 
    const unsigned int* offsets, 
    const unsigned int* sizes
) {
    for (unsigned int i = 0; i < H; i++) {
        unsigned char c = text[i];
        unsigned int offset = offsets[c];

        for (unsigned int j = 0; j < sizes[c]; j++) {
            int k = i - pairs[offset + j].k;

            if (k >= 0 && k < H) {
                R[pairs[offset + j].n * H + k]--;
            }
        }
    }
}

__global__ void mass_search_kernel(
    int* R,
    const char* text, 
    unsigned int H, 
    const Pair* pairs, 
    const unsigned int* offsets, 
    const unsigned int* sizes
) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    if (i >= H) {
        return;
    }
    unsigned char c = text[i];

    unsigned int offset = offsets[c];
    unsigned int size = sizes[c];

    for (unsigned int j = 0; j < size; j++) {
        Pair pair = pairs[offset + j];
        int k = i - pair.k;

        if (k >= 0 && k < H) {
            atomicSub(&R[pair.n * H + k], 1);
        }
    }
}

void mass_search_gpu(
    int* R,
    const char* text, 
    unsigned int H, 
    unsigned int N, 
    const Pair* pairs, 
    unsigned int sum_size, 
    const unsigned int* offsets, 
    const unsigned int* sizes
) {
    char* d_text;
    Pair* d_pairs;
    unsigned int* d_offsets, * d_sizes;
    int* d_R;

    cudaMalloc(&d_text, H);
    cudaMalloc(&d_pairs, sum_size * sizeof(Pair));
    cudaMalloc(&d_offsets, ALPHABET_SIZE * sizeof(int));
    cudaMalloc(&d_sizes, ALPHABET_SIZE * sizeof(int));
    cudaMalloc(&d_R, N * H * sizeof(int));

    cudaMemcpy(d_text, text, H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pairs, pairs, sum_size * sizeof(Pair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, ALPHABET_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes, ALPHABET_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, N * H * sizeof(int), cudaMemcpyHostToDevice);

    unsigned int grid_size = (H + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mass_search_kernel<<<grid_size, BLOCK_SIZE>>>(d_R, d_text, H, d_pairs, d_offsets, d_sizes);
    cudaDeviceSynchronize();

    cudaMemcpy(R, d_R, N * H * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text);
    cudaFree(d_pairs);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
    cudaFree(d_R);
}

bool compare_indexes(int* R1, int* R2, unsigned int N, unsigned int H) {
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < H; j++) {
            if (R1[i * H + j] != R2[i * H + j]) {
                return false;
            }
        }
    }
    return true;
}

struct Params {
    unsigned int min_H;
    unsigned int max_H;
    unsigned int num_H;
    unsigned int N;
    unsigned int min_len;
    unsigned int max_len;
    unsigned int repetitions;
};

Params get_params(unsigned int argc, char** argv) {
    unsigned int min_H = 10000, max_H = 60000, num_H = 6, N = 10000, min_len = 5, max_len = 1000, repetitions = 10;

    for (unsigned int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg.find("--min_H=") == 0) {
            min_H = std::stoi(arg.substr(8));
        }
        else if (arg.find("--max_H=") == 0) {
            max_H = std::stoi(arg.substr(8));
        }
        else if (arg.find("--num_H=") == 0) {
            num_H = std::stoi(arg.substr(8));
        }
        else if (arg.find("--N=") == 0) {
            N = std::stoi(arg.substr(4));
        }
        else if (arg.find("--min_len=") == 0) {
            min_len = std::stoi(arg.substr(10));
        }
        else if (arg.find("--max_len=") == 0) {
            max_len = std::stoi(arg.substr(10));
        }
        else if (arg.find("--rep=") == 0) {
            repetitions = std::stoi(arg.substr(6));
        }
    }

    return { min_H, max_H, num_H, N, min_len, max_len, repetitions };
}

int main(unsigned int argc, char* argv[]) {
    std::cout << std::left
        << std::setw(15) << "H"
        << std::setw(10) << "CPU, ms"
        << std::setw(10) << "GPU, ms"
        << std::setw(10) << "S"
        << std::setw(15) << "compare" << std::endl;

    Params params = get_params(argc, argv);
    unsigned int* Hs = build_Hs(params.min_H, params.max_H, params.num_H);

    std::vector<Pair> table[ALPHABET_SIZE];
    unsigned int offsets[ALPHABET_SIZE];
    unsigned int sizes[ALPHABET_SIZE];

    std::vector<std::string> patterns;
    init_random_patterns(patterns, params.N, params.min_len, params.max_len);
    unsigned int N = patterns.size();
    unsigned int* lens = build_lens(patterns);

    for (unsigned int i = 0; i < params.num_H; i++) {
        double cpu_total_time = 0.0;
        double gpu_total_time = 0.0;

        bool compare_cpu_gpu = true;

        unsigned int H = Hs[i];
        std::string text(H, '\0');
        init_random_text(text, H);

        init_table(table, patterns);
        std::vector<Pair> pairs;
        int max_size = init_pairs(pairs, offsets, sizes, table);

        int* R1 = new int[N * H];
        int* R2 = new int[N * H];

        for (unsigned int rep = 0; rep < params.repetitions; rep++) {
            init_R(R1, lens, N, H);
            init_R(R2, lens, N, H);

            // Вычисление на хосте
            auto cpu_time_start = std::chrono::high_resolution_clock::now();
            mass_search_cpu(R1, text.data(), H, pairs.data(), offsets, sizes);
            auto cpu_time_end = std::chrono::high_resolution_clock::now();
            cpu_total_time += std::chrono::duration<double, std::milli>(cpu_time_end - cpu_time_start).count();

            // Вычисление на девайсе
            auto gpu_time_start = std::chrono::high_resolution_clock::now();
            mass_search_gpu(R2, text.data(), H, N, pairs.data(), pairs.size(), offsets, sizes);
            auto gpu_time_end = std::chrono::high_resolution_clock::now();
            gpu_total_time += std::chrono::duration<double, std::milli>(gpu_time_end - gpu_time_start).count();

            compare_cpu_gpu &= compare_indexes(R1, R2, N, H);
        }

        delete[] R1;
        delete[] R2;

        double cpu_avg_time = cpu_total_time / params.repetitions;
        double gpu_avg_time = gpu_total_time / params.repetitions;
        double S = cpu_avg_time / gpu_avg_time;

        std::cout << std::fixed << std::setprecision(2) << std::left 
            << std::setw(15) << H
            << std::setw(10) << cpu_avg_time
            << std::setw(10) << gpu_avg_time
            << std::setw(10) << S
            << std::setw(15) << (compare_cpu_gpu ? "Equal" : "Not equal") << std::endl;
    }

    delete[] lens;
}