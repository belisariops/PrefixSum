//
// Created by belisariops on 11/25/17.
//

#ifndef PREFIXSUM_CPUPREFIXSUM_H
#define PREFIXSUM_CPUPREFIXSUM_H

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
#endif

#define num_threads __cilkrts_get_nworkers()

class CpuPrefixSum {
    void prefix_sum_block(uint* A, uint size);
    void prefix_sum_seq(uint *A, int n);
    void prefix_sum_iter(uint *A, unsigned int n);
    void prefix_sum_tree(uint *A, int n);
    void prefix_sum_not_work_efficient(uint* x, uint *y, const uint numValues);

private:
    void upsweep(uint *A, uint ll, uint ul);
    void downsweep(uint *A, uint ll, uint ul, int n);
    void scanNotWorkEfficient(uint *array,uint* read,int index, int iteration);



};


#endif //PREFIXSUM_CPUPREFIXSUM_H
