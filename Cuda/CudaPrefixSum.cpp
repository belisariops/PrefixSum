//
// Created by belisariops on 11/24/17.
//

#include "CudaPrefixSum.h"

extern void runCuda(int *A, int size) ;
extern void initCuda(int size);
extern void destroyCuda();

CudaPrefixSum::CudaPrefixSum(int size) {
    this->arraySize = size;
    initCuda(size);
}

CudaPrefixSum::~CudaPrefixSum() {
    destroyCuda();
}

int *CudaPrefixSum::run(int *A) {
    runCuda(A, arraySize);
    return 0;
}
