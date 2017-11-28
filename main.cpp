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
//#include "OpenCL/OpenCLPrefixSum.h"

#endif

#define num_threads __cilkrts_get_nworkers()


int main() {
    uint numValues;
    const unsigned int numIterations = 5;
    std::ofstream myfile;
    std::string fileName = "cpu_prefix_results.txt";
    myfile.open (fileName);
    myfile << "NumValues     " <<"Secuencial     " << "Paralelo" << std::endl;
    std::chrono::duration<double> duration;
    CpuPrefixSum cpu = CpuPrefixSum();
    for (int i = 3; i < 31; ++i) {
        int value = (1 << i);
        numValues =  (uint)value;
        uint *x = (uint *)malloc(sizeof(uint) * numValues);
        uint *y = (uint *)malloc(sizeof(uint) * numValues);

        for (uint l = 0; l < numValues; ++l) {
            x[l] = l;
            y[l] = l;
        }

        auto start = std::chrono::system_clock::now();
        cpu.prefix_sum_seq(x, numValues);
        auto end = std::chrono::system_clock::now();
        duration = end-start;
        myfile << value <<"    " << duration.count() << "    ";
        start = std::chrono::system_clock::now();
        cpu.prefix_sum_block(y, numValues);
        end = std::chrono::system_clock::now();
        duration = end-start;
        myfile << duration.count() << std::endl;

        free(x);
        free(y);
    }
    myfile.close();


    return 0;
}