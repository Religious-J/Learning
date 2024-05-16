#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>
__global__ void warpReduction(int *data, int size) {
  int tid = threadIdx.x;
  int warpSize = 32; // Assuming a warp size of 32
  // Warp-level reduction
  for (int s = blockDim.x / 2; s > warpSize; s >>= 1) {
    if (tid < s) {
      for (int dim = 0; dim < 3; ++dim) {
        data[tid * 3 + dim] += data[(tid + s) * 3 + dim];
      }
    }
    __syncwarp();
  }
  // Final reduction within the last warp
  if (tid < warpSize) {
    for (int s = warpSize / 2; s > 0; s >>= 1) {
      for (int dim = 0; dim < 3; ++dim) {
        data[tid * 3 + dim] += data[(tid + s) * 3 + dim];
      }
      __syncwarp();
    }
  }
}
__global__ void blockReduction(int *data, int size) {
  int tid = threadIdx.x;
  int blockSize = blockDim.x;
  // Block-level reduction
  for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      for (int dim = 0; dim < 3; ++dim) {
        data[tid * 3 + dim] += data[(tid + s) * 3 + dim];
      }
    }
    __syncthreads();
  }
}
int main() {
  int size = 32; // Size of the data array
  int *data_warp;
  int *data_block;
  int *warm;

  cudaMallocManaged(&data_warp, size * 3 * sizeof(int));
  cudaMallocManaged(&data_block, size * 3 * sizeof(int));
  cudaMallocManaged(&warm, size * 3 * sizeof(int));

  // Initialize data arrays
  for (int i = 0; i < size * 3; ++i) {
    data_warp[i] = i + 1;
    data_block[i] = i + 1;
    warm[i] = i+1;
  }
  // Launch the kernel for warp-level reduction
  int blockSize = 32; // Assuming a block size of 32
  int numBlocks = 1;
  // Start timer for warp-level reduction

  // warm
  warpReduction<<<numBlocks, blockSize>>>(warm, size);

  cudaDeviceSynchronize();

  struct timeval start_warp, end_warp;
  gettimeofday(&start_warp, NULL);
  warpReduction<<<numBlocks, blockSize>>>(data_warp, size);
  // Wait for the kernel to finish
  cudaDeviceSynchronize();
  // Stop timer for warp-level reduction
  gettimeofday(&end_warp, NULL);
  float elapsedTime_warp = (end_warp.tv_sec - start_warp.tv_sec) * 1000.0f +
                           (end_warp.tv_usec - start_warp.tv_usec) / 1000.0f;
  printf("Elapsed time for warp-level reduction: %.3f ms\n", elapsedTime_warp);
  // Launch the kernel for block-level reduction
  // Reset data array for block-level reduction
  // for (int i = 0; i < size * 3; ++i) {
  //   data_block[i] = i + 1;
  // }
  // Start timer for block-level reduction
  struct timeval start_block, end_block;
  gettimeofday(&start_block, NULL);
  blockReduction<<<numBlocks, blockSize>>>(data_block, size);
  // Wait for the kernel to finish
  cudaDeviceSynchronize();
  // Stop timer for block-level reduction
  gettimeofday(&end_block, NULL);
  float elapsedTime_block = (end_block.tv_sec - start_block.tv_sec) * 1000.0f +
                            (end_block.tv_usec - start_block.tv_usec) / 1000.0f;
  printf("Elapsed time for block-level reduction: %.3f ms\n",
         elapsedTime_block);
  // Free memory
  cudaFree(data_warp);
  cudaFree(data_block);
  cudaFree(warm);
  return 0;
}
