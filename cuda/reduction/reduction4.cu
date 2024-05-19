#include <thread>
__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];

  unsigned int tid = threadIdx.x;
  // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  //  sdata[tid] = g_idata[i];
  //  with two loads and first add of the reduction
  //  perform first level of reduction
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >> 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}
