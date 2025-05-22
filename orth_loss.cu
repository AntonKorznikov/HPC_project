#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>





__global__ void shuffle_rows(const float* W_dec, const int* perm, float* shuffled, int D, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D * N) {
        int i = idx / N;  
        int j = idx % N;  /
        int src_row = perm[i];
        shuffled[i * N + j] = W_dec[src_row * N + j];
    }
}


__global__ void compute_max_sqr(const float* cos_sims, float* max_vals, int group_count, int group_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < group_count * group_size) {
        int k = idx / group_size;  
        int i = idx % group_size; 
        float max_val = -1.0f;
        for (int j = 0; j < group_size; j++) {
            if (j != i) { 
                float val = cos_sims[k * group_size * group_size + i + j * group_size];
                if (val > max_val) max_val = val;
            }
        }
        max_vals[idx] = max_val * max_val;
    }
}








int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);


    std::vector<std::pair<int, int>> sizes = {{1024, 256}, {2048, 512}, {4096, 1024}, 
                                             {8192, 2048}, {16384, 4096}, {32768, 8192}};

    std::ofstream out_file("cuda_times.txt");
    for (auto [D, N] : sizes) {
        int group_count = 32;
        int group_size = D / group_count;

     
        std::vector<float> W_dec_host(D * N);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0, 1.0);
        for (auto& val : W_dec_host) {
            val = distribution(generator);
        }


        std::vector<int> perm(D);
        for (int i = 0; i < D; i++) perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), generator);


        float *W_dec_dev, *shuffled_dev, *cos_sims_dev, *max_vals_dev;
        int* perm_dev;
        cudaMalloc(&W_dec_dev, D * N * sizeof(float));
        cudaMalloc(&perm_dev, D * sizeof(int));

        cudaMalloc(&shuffled_dev, D * N * sizeof(float));


        cudaMalloc(&cos_sims_dev, group_count * group_size * group_size * sizeof(float));
        cudaMalloc(&max_vals_dev, group_count * group_size * sizeof(float));


        cudaMemcpy(W_dec_dev, W_dec_host.data(), D * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(perm_dev, perm.data(), D * sizeof(int), cudaMemcpyHostToDevice);


        int threads = 256;
        int blocks = (D * N + threads - 1) / threads;
        shuffle_rows<<<blocks, threads>>>(W_dec_dev, perm_dev, shuffled_dev, D, N);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  group_size, group_size, N,
                                  &alpha,
                                  shuffled_dev, N, group_size * N,
                                  shuffled_dev, N, group_size * N,
                                  &beta,
                                  cos_sims_dev, group_size, group_size * group_size,
                                  group_count);


        blocks = (group_count * group_size + threads - 1) / threads;
        compute_max_sqr<<<blocks, threads>>>(cos_sims_dev, max_vals_dev, group_count, group_size);
        cudaDeviceSynchronize();


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        shuffle_rows<<<blocks, threads>>>(W_dec_dev, perm_dev, shuffled_dev, D, N);

        cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  group_size, group_size, N,
                                  &alpha,
                                  shuffled_dev, N, group_size * N,
                                  shuffled_dev, N, group_size * N,
                                  &beta,
                                  cos_sims_dev, group_size, group_size * group_size,
                                  group_count);
        compute_max_sqr<<<blocks, threads>>>(cos_sims_dev, max_vals_dev, group_count, group_size);


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);


        std::string result = "Size " + std::to_string(D) + "x" + std::to_string(N) + ": " + std::to_string(milliseconds) + " ms\n";
        std::cout << result;
        out_file << result;
        

        cudaFree(W_dec_dev);
        cudaFree(perm_dev);
        cudaFree(shuffled_dev);
        cudaFree(cos_sims_dev);
        cudaFree(max_vals_dev);
        cudaEventDestroy(start);


        cudaEventDestroy(stop);
    }

    out_file.close();
    cublasDestroy(handle);
    return 0;
}