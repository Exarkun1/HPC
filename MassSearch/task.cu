#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

const size_t ALPHABET_SIZE = 256;
const size_t BLOCK_SIZE = 256;

struct Pair {
    unsigned int n;
    unsigned int k;
};

void init_table(std::vector<Pair>* table, const std::vector<std::string>& patterns) {
    for (unsigned int i = 0; i < patterns.size(); i++) {
        for (unsigned int k = 0; k < patterns[i].size(); k++) {
            unsigned char c = patterns[i][k];
            table[c].push_back({i, k});
        }
    }
}

void init_lens(unsigned int* lens, const std::vector<std::string>& patterns) {
    for (unsigned int i = 0; i < patterns.size(); i++) {
        lens[i] = patterns[i].size();
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
    const char* text, 
    unsigned int H, 
    const Pair* pairs, 
    const unsigned int* offsets, 
    const unsigned int* sizes, 
    int* R
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
    const char* text, 
    unsigned int H, 
    const Pair* pairs, 
    const unsigned int* offsets, 
    const unsigned int* sizes, 
    int* R
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
    const char* text, 
    unsigned int H, 
    unsigned int N, 
    const Pair* pairs, 
    unsigned int sum_size, 
    const unsigned int* offsets, 
    const unsigned int* sizes, 
    int* R
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

    mass_search_kernel<<<grid_size, BLOCK_SIZE>>>(d_text, H, d_pairs, d_offsets, d_sizes, d_R);
    cudaDeviceSynchronize();

    cudaMemcpy(R, d_R, N * H * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_text);
    cudaFree(d_pairs);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
    cudaFree(d_R);
}

int main() {
    std::string text = "abracadabra";
    size_t H = text.size();

    std::vector<std::string> patterns = {"abra", "cad"};
    size_t N = patterns.size();

    unsigned int* lens = new unsigned int[N];
    init_lens(lens, patterns);

    std::vector<Pair> table[ALPHABET_SIZE];
    init_table(table, patterns);

    std::vector<Pair> pairs;
    unsigned int offsets[ALPHABET_SIZE];
    unsigned int sizes[ALPHABET_SIZE];
    int max_size = init_pairs(pairs, offsets, sizes, table);

    int* R = new int[N * H];
    init_R(R, lens, N, H);

    mass_search_cpu(text.data(), H, pairs.data(), offsets, sizes, R);

    std::cout << "CPU\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < H; j++) {
            if (R[i * H + j] == 0) {
                std::cout << "Pattern \"" << patterns[i]
                          << "\" found at position " << j << "\n";
            }
        }
    }

    init_R(R, lens, N, H);

    mass_search_gpu(text.data(), H, N, pairs.data(), pairs.size(), offsets, sizes, R);

    std::cout << "GPU\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < H; j++) {
            if (R[i * H + j] == 0) {
                std::cout << "Pattern \"" << patterns[i]
                          << "\" found at position " << j << "\n";
            }
        }
    }

    delete[] lens;
    delete[] R;
}