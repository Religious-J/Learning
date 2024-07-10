#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>


template <typename T>
// warp（32）sync
__device__ void warpReduce(volatile T *sdata, int tid) {   
  sdata[tid] += sdata[tid + 32];    
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

template <typename T>
__global__ void vector_div_vector_dot_kernel(
    const int size,
    const T* hphi,
    const T* prec,
    const T* sphi,
    T* grad,
    T* result_dot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ T sdata[256]; 
    T temp_value = 0.0;

    if (i < size)
    {
        grad[i] = hphi[i] / prec[i];
        temp_value = grad[i] * sphi[i];
    }

    sdata[tid] = temp_value;
    __syncthreads();

    // reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // warp
    if (tid < 32)
        warpReduce(sdata, tid);

    // Write the partial sum of each block to global memory
    if (threadIdx.x == 0) {
        atomicAdd(result_dot, sdata[0]);
    }
}

template <typename T>
int main() {
    int N = 1024;

    // 初始化主机矢量
    std::vector<T> A(N, 1.0);
    std::vector<T> B(N, 2.0);
    std::vector<T> C(N, 3.0);
    std::vector<T> Temp(N, 0.0);
    T result_dot = 0.0;

    // 分配设备矢量
    T *d_A, *d_B, *d_C, *d_Temp, *d_result_dot;
    cudaMalloc((void**)&d_A, A.size() * sizeof(T));
    cudaMalloc((void**)&d_B, B.size() * sizeof(T));
    cudaMalloc((void**)&d_C, C.size() * sizeof(T));
    cudaMalloc((void**)&d_Temp, Temp.size() * sizeof(T));
    cudaMalloc((void**)&d_result_dot, sizeof(T));

    // 复制数据到设备
    cudaMemcpy(d_A, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), C.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_dot, &result_dot, sizeof(T), cudaMemcpyHostToDevice);

    // 配置CUDA内核
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 调用CUDA内核
    vector_div_vector_dot_kernel<<<numBlocks, blockSize>>>(N, d_A, d_B, d_C, d_Temp, d_result_dot);

    // 复制结果回主机
    cudaMemcpy(&result_dot, d_result_dot, sizeof(T), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Dot product = " << result_dot << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Temp);
    cudaFree(d_result_dot);

    return 0;
}