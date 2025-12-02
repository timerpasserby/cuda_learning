
#include <cmath>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
        }                                                                \
    }

// Step 3: Define add kernel
/**
 * @brief CUDA kernel for element-wise addition: c = a+b
 * @tparam T The data type of the arrays, which can be any type that supports addition operations(e.g.. int, float)
 *
 * @param c Pointer to the result array, where the results of the addition are stored.
 * @param a Pointer to the first input array.
 * @param b Pointer to the second input array.
 * @param n The number of elements in the arrays. The arrays are assumed to be of equal length.
 */
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    // Step 1: Prepare & initialize data
    constexpr size_t N = 1 << 20; // ~1M elements
    constexpr size_t size_bytes = sizeof(float) * N;

    // Initialize data
    const std::vector<float> h_a(N, 1);
    const std::vector<float> h_b(N, 2);
    std::vector<float> h_c(N, 0);

    // Step 2: Allocate device memory & transfer to global memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_c, size_bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), size_bytes, cudaMemcpyHostToDevice));

    // Step 4: Call the cpu addition function
    // Set up kernel configuration
    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);

    // Call cuda add kernel
    add_kernel<<<grid_dim, block_dim>>>(d_c, d_a, d_b, N);

    // Step 5: Transfer data from global mem to host mem
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, size_bytes, cudaMemcpyDeviceToHost));

    // Step 6: Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(h_c[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    if (d_a)
    {
        CUDA_CHECK(cudaFree(d_a));
    }
    if (d_b)
    {
        CUDA_CHECK(cudaFree(d_b));
    }
    if (d_c)
    {
        CUDA_CHECK(cudaFree(d_c));
    }
}