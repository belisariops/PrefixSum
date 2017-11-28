//
// Created by belisariops on 11/27/17.
//

#ifndef PREFIXSUM_OPENCLPREFIXSUM_H
#define PREFIXSUM_OPENCLPREFIXSUM_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

class OpenCLPrefixSum {
public:
    explicit OpenCLPrefixSum(int size);
    ~OpenCLPrefixSum();
    void runNaiveSum(int *A, int size);
    void runPrefixSum(int *A, int size);

private:
    int N_ELEMENTS;
    int platform_id=0;
    int device_id=0;
    cl::Kernel kernel_1;
    cl::Kernel kernel_2;
    cl::Kernel kernel_3;
    cl::Buffer bufferA;
    cl::Buffer bufferB;
    cl::CommandQueue queue;
};


#endif //PREFIXSUM_OPENCLPREFIXSUM_H
