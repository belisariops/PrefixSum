#include <iostream>
#include <thread>
#include <cmath>

#define THREADS 8
//
//#ifdef NOPARALLEL
//#define __cilkrts_get_nworkers() 1
//#define cilk_for for
//#define cilk_spawn
//#define cilk_sync
//#else
//#include <cilk/cilk.h>
//#include <cilk/cilk_api.h>
//#include <cilk/common.h>
#include <fstream>
//#include "CPU/CpuPrefixSum.h"
#include "Cuda/CudaPrefixSum.h"
#include "OpenCL/OpenCLPrefixSum.h"

//#endif

//#define num_threads __cilkrts_get_nworkers()


int main() {
    uint numValues;
    const unsigned int numIterations = 5;
    std::ofstream myfile;
    std::string fileName = "cpu_block";
    myfile.open (fileName);
    myfile << "CPUseq " << "CPUmult" << "CUDA " << "OpenCL";

    for (int i = 3; i < 4; ++i) {
        CudaPrefixSum cuda = CudaPrefixSum(1 << 20);
        OpenCLPrefixSum opencl = OpenCLPrefixSum(1 << 20);
        numValues =  (uint)(1 << 20);
        int *x = (int *)malloc(sizeof(int) * numValues);
        int *y = (int *)malloc(sizeof(int) * numValues);

        for (uint l = 0; l < numValues; ++l) {
            x[l] = l;
            y[l] = l;
        }
        for (int j = 0; j < 1; ++j) {
            auto start = std::chrono::system_clock::now();
            opencl.runPrefixSum(x, 1 << 20);
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end-start;

        }

        for (int k = 0; k < 32; ++k) {
            std::cout << x[k] <<" ";
        }
        std::cout << std::endl;

        int a[32];
        for (int m = 0; m < 32; ++m) {
            a[m] = m;
        }

        for (int n = 1; n < 32; ++n) {
            a[n] += a[n-1];
        }

        for (int i1 = 0; i1 < 32; ++i1) {
            std::cout << a[i1] <<" ";
        }
        std::cout << std::endl;
        free(x);
        free(y);
    }


    return 0;
}