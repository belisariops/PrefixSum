//
// Created by belisariops on 11/24/17.
//

#include "CudaPrefixSum.h"

extern void runNaiveCuda(int *A, int size);
extern void runPrefixCuda(int *A, int size);
extern void initCuda(int size);
extern void destroyCuda();

CudaPrefixSum::CudaPrefixSum(int size) {
    this->arraySize = size;
    initCuda(size);
}

CudaPrefixSum::~CudaPrefixSum() {
    destroyCuda();
}

int *CudaPrefixSum::runNaive(int *A) {
    runNaiveCuda(A, arraySize);
    return A;
}

int *CudaPrefixSum :: runSum(int *A) {
    runPrefixCuda(A, arraySize);
    return A;
}
