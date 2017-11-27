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


__global__ void naivePrefixSum(int *A, int *B, int size, int iteration) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        if (index >= (1 << (iteration - 1)))
            A[index] = B[(int)(index - (1 << (iteration - 1)))] + B[index];
        else
            A[index] = B[index];
//        int aux = A[index];
//        A[index] = B[index];
//        B[index] = aux;
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

void runCuda(int *A, int size) {

    // Size, in bytes, of each vector
    size_t bytes = size*sizeof(int);


    // Copy host vectors to device
    cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, A, bytes, cudaMemcpyHostToDevice);


//    int blockSize, gridSize, n;

//    // Tamaño de la matriz.
//    n = height*width;
//
//    // Tamaño del bloque. Elegir entre 32 y 31.
//    //blockSize = 32;
//    blockSize = 32;
//
//    // Number of thread blocks in grid
//    gridSize = (int)ceil((float)n/blockSize);
    int *aux;
    // Execute the kernel
    for (int i = 1; i <= (int)log2(size); ++i) {
        naivePrefixSum<<< size, 1 >>>(d_a, d_b, size, i);
        aux = d_b;
        d_b = d_a;
        d_a = aux;
    }

    // Copy array back to host
    cudaMemcpy( A, d_b, bytes, cudaMemcpyDeviceToHost );




}