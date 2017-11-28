//
// Created by belisariops on 11/25/17.
//

#include "CpuPrefixSum.h"

void CpuPrefixSum::prefix_sum_block(uint *A, uint size) {
    uint chk = size/num_threads;
    cilk_for(uint i = 0; i < num_threads; i++) {
        uint  ll = i*chk, ul = ll + chk;
        if(i == num_threads-1)
            ul = size;

        uint acc = 0;
        for(uint j = ll; j < ul; j++) {
            A[j] += acc;
            acc = A[j];
        }
    }
    for(uint i = 1; i < num_threads-1; i++)
        A[((i+1)*chk)-1] += A[i*chk-1];

    if(num_threads > 1)
        A[size-1] += A[(num_threads-1)*chk-1];

    cilk_for(uint i = 1; i < num_threads; i++) {
        uint ll = i*chk, ul = ll + chk - 1;
        if(i == num_threads-1)
            ul = size-1;

        uint acc = A[ll-1];
        for(uint j = ll; j < ul; j++) {
            A[j] += acc;
        }
    }
}

void CpuPrefixSum::prefix_sum_seq(uint *A, int n) {
    for(int i = 1; i < n; i++) {
        A[i] += A[i-1];
    }
}

void CpuPrefixSum::prefix_sum_iter(uint *A, unsigned int n) {
    if (n == 1){
        return;
    }
    int itr;
    int levels = log2(n);

    // Up-sweep
    for (itr = 1; itr <= levels; itr++) {
        int desp = 1 << itr;

        cilk_for(unsigned int i = desp-1; i < n; i += desp){
            A[i] = A[i] + A[i - (1 << (itr - 1))];
        }
    }

    // Down-sweep
    for (itr = levels-1; itr > 0; itr--) {
        int desp = 1 << itr;

        cilk_for(unsigned int i = desp-1; i < n-1; i += desp) {
            int idx = i + (1 << (itr-1));
            A[idx] += A[i];
        }
    }
}

void CpuPrefixSum::prefix_sum_tree(uint *A, int n) {
    upsweep(A, 0, n-1);
    downsweep(A, 0, n-1, n);
}

void CpuPrefixSum::prefix_sum_not_work_efficient(uint *x, uint *y, const uint numValues) {
    uint *aux;
    const int iterations = (int)log2(numValues);
    int a = 0;
    for (int d = 1; d <= iterations; ++d) {
        cilk_for(int i=0; i < THREADS; i++) {
            for (int j = i*numValues/THREADS; j < (i+1)*numValues/THREADS; ++j) {
                scanNotWorkEfficient(x,y,j,d);
            }
        }
        cilk_for(int j= THREADS * (numValues/THREADS); j < numValues; j++)
        scanNotWorkEfficient(x,y,j,d);
        aux = x;
        x = y;
        y = aux;
    }

}

void CpuPrefixSum::upsweep(uint *A, uint ll, uint ul) {
    if(ul - ll == 1) {
        A[ul] += A[ll];
        return;
    }
    uint k = pow(2,((int)log2(ul-ll+1)-1));

    cilk_spawn upsweep(A, ll, ll+k-1);
    upsweep(A, ll+k, ul);
    cilk_sync;
    A[ul] += A[ul-k];
}

void CpuPrefixSum::downsweep(uint *A, uint ll, uint ul, int n) {
    if(ul - ll == 0) {
        return;
    }

    uint k = pow(2,((int)log2(ul-ll+1)-1));

    if(ul+k < n)
        A[ul+k] += A[ul];

    cilk_spawn downsweep(A, ll, ll+k-1, n);
    downsweep(A, ll+k, ul, n);
    // Implicit cilk_sync
}

void CpuPrefixSum::scanNotWorkEfficient(uint *array, uint *read, int index, int iteration) {
    if (index >= (1 << (iteration - 1)))
        array[index] = read[(int)(index - (1 << (iteration - 1)))] + read[index];
    else
        array[index] = read[index];
}
