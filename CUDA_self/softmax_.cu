#include <iostream>

void softmax_forward_cpu(float *out, const float *inp, int N, int C)
{
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++)
    {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++)
        {
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++)
        {
            out_row[j] *= norm;
        }
    }
}

void softmax_forward_online_cpu(float *out, const float *inp, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++)
        {
            float maxval_prev = maxval;
            // 最大值发生变化时，才需要调整历史值，完成增程式更新
            if (inp_row[j] > maxval)
            {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            }
            else
            {
                // 否则，直接加上增值即可
                sum += expf(inp_row[j] - maxval);
            }
        }
    }
}

__global__ void softmax_forward_kernel1(float *out, const float *inp, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++)
        {
            maxval = std::max(maxval, inp_row[j]);
        }

        double sum = 0.0;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        for (int j = 0; j < C; j++)
        {
            out_row[j] /= (float)sum;
        }
    }
}

__global__ void softmax_forward_kernel2(float *out, const float *inp, int N, int C)
{
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // range[0, N)
    int tid = threadIdx.x; // range[0, block_size)
    int block_size = blockDim.x;
    const float *x = inp + idx * C;

    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;

    __syncthreads();

    // reductions;

    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    // thread coarsening again , for the sum;
    x = out + idx * c;
    float sumval = 0.0f;
    __syncthreads();

    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthread();
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] = x[i] / sum;
    }
}