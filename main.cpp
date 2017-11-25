#include <iostream>
#include <thread>
#include <cmath>

#define THREADS 8

#ifdef NOPARALLEL
#define __cilkrts_get_nworkers() 1
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#else
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/common.h>
#include <fstream>
#include "CPU/CpuPrefixSum.h"

#endif

#define num_threads __cilkrts_get_nworkers()


int main() {
    uint numValues;
    const unsigned int numIterations = 5;
    CpuPrefixSum cpu = CpuPrefixSum();
    auto start,end;
    std::ofstream myfile;
    std::string fileName = "cpu_block";
    myfile.open (fileName);
    myfile << "CPUseq " << "CPUmult" << "CUDA " << "OpenCL";

    for (int i = 5; i < 31; ++i) {
        numValues =  (uint)(1 << i);
        uint *x = (uint *)malloc(sizeof(uint) * numValues);
        uint *y = (uint *)malloc(sizeof(uint) * numValues);

        for (uint l = 0; l < numValues; ++l) {
            x[l] = l;
            y[l] = l;
        }
        for (int j = 0; j < numIterations; ++j) {
            start = std::chrono::system_clock::now();
            cpu.prefix_sum_block(x, numValues);
            end = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end-start;

        }

        free(x);
        free(y);
    }


    std::cout << duration.count() << std::endl;
    return 0;
}