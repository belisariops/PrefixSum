//
// Created by belisariops on 11/24/17.
//

#ifndef PREFIXSUM_CUDAPREFIXSUM_H
#define PREFIXSUM_CUDAPREFIXSUM_H


class CudaPrefixSum {
public:
    CudaPrefixSum(int size);
    ~CudaPrefixSum();
    int *runNaive(int *A);
    int *runSum(int *A);
private:
    int arraySize;
};


#endif //PREFIXSUM_CUDAPREFIXSUM_H
