#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cstdio>


// Device input vectors
int *d_a;
//Device output vector
int *d_b;


__device__ int mod(int a, int b) {
    return a >= 0 ? a%b :  ( b - abs ( a%b ) ) % b;
}


__global__ void update(int *A, int *B, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {


    }
}

void initCuda(int size) {
    // Allocate memory for each vector on GPU
    cudaMalloc((void **) &d_a, size*sizeof(int));
    cudaMalloc((void **) &d_b, size*sizeof(int));
}

void destroyCuda() {
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);

}

void updateCuda(int *A, int size) {

    // Size, in bytes, of each vector
    size_t bytes = size*sizeof(int);


    // Copy host vectors to device
    cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);


    int blockSize, gridSize, n;

//    // Tamaño de la matriz.
//    n = height*width;
//
//    // Tamaño del bloque. Elegir entre 32 y 31.
//    //blockSize = 32;
//    blockSize = 32;
//
//    // Number of thread blocks in grid
//    gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    update<<< size, 1 >>>(d_a, d_b, size);

    // Copy array back to host
    cudaMemcpy( A, d_b, bytes, cudaMemcpyDeviceToHost );




}