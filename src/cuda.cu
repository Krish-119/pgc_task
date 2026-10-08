#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    long long N)
{
    long long i =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char* argv[])
{
    long long N = 100000000;

    if (argc >= 2)
        N = atoll(argv[1]);

    size_t bytes = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (long long i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(
        d_A, h_A, bytes,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B, h_B, bytes,
        cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    int blocks =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd<<<blocks, threadsPerBlock>>>(
        d_A, d_B, d_C, N);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    cudaMemcpy(
        h_C, d_C, bytes,
        cudaMemcpyDeviceToHost);

    double seconds = milliseconds / 1000.0;

    cout << endl;
    cout << "========================================" << endl;
    cout << "       CPU VS GPU DATA PROCESSING" << endl;
    cout << "========================================" << endl;
    cout << "Method              : CUDA GPU" << endl;
    cout << "Device              : NVIDIA RTX 3050" << endl;
    cout << "Data Size           : " << N << " elements" << endl;
    cout << "GPU Threads/Block   : " << threadsPerBlock << endl;
    cout << "GPU Blocks          : " << blocks << endl;
    cout << "Execution Time      : " << seconds << " seconds" << endl;
    cout << "Execution Time      : " << milliseconds << " ms" << endl;
    cout << "Result Check        : " << h_C[N - 1] << endl;
    cout << "========================================" << endl;
    cout << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}