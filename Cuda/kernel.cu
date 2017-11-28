#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cstdio>


// Device input vectors
int *d_a;
//Device output vector
int *d_b;


__global__ void naivePrefixSum(int *A, int *B, int size, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        for (int i = 1; i <= iterations; ++i) {
            if (index >= (1 << (i - 1)))
                A[index] = B[(int) (index - (1 << (i - 1)))] + B[index];
            else
                A[index] = B[index];

            __syncthreads();

            int aux = A[index];
            A[index] = B[index];
            B[index] = aux;

            __syncthreads();

        }
    }
}

__global__ void upSweep(int *A, int size, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        for (int i = 0; i >= iterations ; ++i) {
            if ((index + 1) % (1 << (i + 1)))
                A[index] = A[index - (1<<i)] + A[index];
            __syncthreads();
        }
    }

}


__global__ void downSweep(int *A, int size, int iterations) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        int aux;
        for (int i = iterations; i >= 0 ; --i) {
            if ((index + 1) % (1 << (i + 1))) {
                aux = A[index];
                A[index] = A[index - (1<<i)] + A[index];
                A[index - (1<<i)] = aux;
            }
            __syncthreads();
        }
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

void runPrefixCuda(int *A, int size) {

    // Size, in bytes, of each vector
    size_t bytes = size*sizeof(int);


    // Copy host vectors to device
    cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
    // Execute the kernels
    upSweep<<< size, 1 >>>(d_a, size, (int)(log2(size) - 1));
    downSweep<<< size, 1 >>>(d_a, size, (int)(log2(size) - 1));


    // Copy array back to host
    cudaMemcpy( A, d_b, bytes, cudaMemcpyDeviceToHost );

}

void runNaiveCuda(int *A, int size) {

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
    naivePrefixSum<<< size, 1 >>>(d_a, d_b, size, (int)log2(size));


    // Copy array back to host
    cudaMemcpy( A, d_b, bytes, cudaMemcpyDeviceToHost );




}