#include <bits/stdc++.h>
#include <cuda.h>
#include <sys/time.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;
#define THREAD_PER_BLOCK 256

template <typename DType>
__global__ void reduce(DType* d_in, DType* d_out) {
  //  为每一个block计算起始地址，设置索引来计算
  DType* input_begin = d_in + blockIdx.x * blockDim.x;
  for (int i = 1; i < blockDim.x; i *= 2) {
    if (threadIdx.x % (2 * i) == 0 && (threadIdx.x + i) < blockDim.x) {
      input_begin[threadIdx.x] += input_begin[threadIdx.x + i];
    }
    // 进行同步，确保所有线程都完成计算
    __syncthreads();
  }
  // write result for this block to global memory
  if (threadIdx.x == 0) {
    d_out[blockIdx.x] = input_begin[0];
  }
}
template <typename DType>
__global__ void reduce01(DType* d_in, DType* d_out) {
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0 && (tid + s) < blockDim.x) {
      d_in[i] += d_in[i + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[i];
  }
}

template <typename DType>
bool check(DType* d_out, DType* res, int n) {
  for (int i = 0; i < n; ++i) {
    //  浮点数计算总会有差异，给一个容差范围
    if (abs(d_out[i] - res[i]) > 0.005) {
      return false;
    }
  }
  return true;
}

int main() {
  const int N = 32 * 1024 * 1024;
  float* h_a = (float*)malloc(N * sizeof(float));
  float* d_a;
  cudaMalloc((void**)&d_a, N * sizeof(float));

  int block_num = N / THREAD_PER_BLOCK;
  float* h_out = (float*)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
  float* d_out;
  cudaMalloc((void**)&d_out, (N / THREAD_PER_BLOCK) * sizeof(float));
  float* res = (float*)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
  //  set input values
  for (int i = 0; i < N; i++) {
    h_a[i] = 2.0 * (float)drand48() - 1.0;
  }

  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
      cur += h_a[i * THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  dim3 Grid(N / THREAD_PER_BLOCK, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  reduce01<<<Grid, Block>>>(d_a, d_out);

  cudaMemcpy(h_out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);

  if (check(h_out, res, block_num)) {
    cout << "the ans is right" << endl;
  } else {
    printf("the ans is wrong\n");
    for (int i = 0; i < block_num; i++) {
      printf("%lf ", h_out[i]);
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_out);
}