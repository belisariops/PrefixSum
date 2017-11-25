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

void prefix_sum_block(uint* A, uint size) {
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

/* Sequential prefix sum */
void prefix_sum_seq(uint *A, int n) {
    for(int i = 1; i < n; i++) {
        A[i] += A[i-1];
    }
}

/* Assuming n is a power of two */
/* To Do: Implement the version for any n */
void prefix_sum_iter(uint *A, unsigned int n) {
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

void upsweep(uint *A, uint ll, uint ul) {
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

void downsweep(uint *A, uint ll, uint ul, int n) {
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

/* Assuming n is a power of two */
/* To Do: Implement the version for any n */
void prefix_sum_tree(uint *A, int n) {
    upsweep(A, 0, n-1);
    downsweep(A, 0, n-1, n);
}

void scanNotWorkEfficient(uint *array,uint* read,int index, int iteration) {
    if (index >= (1 << (iteration - 1)))
        array[index] = read[(int)(index - (1 << (iteration - 1)))] + read[index];
    else
        array[index] = read[index];
}

void parallelPrefix(uint* x, uint *y, const uint numValues) {
    uint *aux;
//
//    std::thread threads[numValues];
//    int joinCount;
    const int iterations = (int)log2(numValues);
//    for (int d = 1; d <= iterations; ++d) {
//        joinCount = 0;
//        threads[0] = std::thread(scanNotWorkEfficient, x, y, 0, d);
//        for (int i = 1; i < numValues; ++i) {
//            threads[i] = std::thread(scanNotWorkEfficient, x, y, i, d);
//            if (!((i + 1) % THREADS))
//                for (int j = THREADS - 1; j >= 0; --j) {
//                    threads[i - j].join();
//                    joinCount++;
//                }
//
//        }
//        while (joinCount < numValues) {
//            threads[joinCount++].join();
//        }
//        aux = x;
//        x = y;
//        y = aux;
//
//    }
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

int main() {
    const uint numValues = 1 << 30;

    auto start = std::chrono::system_clock::now();

    uint *x = (uint *)malloc(sizeof(uint) * numValues);
    uint *y = (uint *)malloc(sizeof(uint) * numValues);

    for (int l = 0; l < numValues; ++l) {
        x[l] = l;
        y[l] = l;
    }
//    prefix_sum_block(x, numValues);
//    prefix_sum_iter(x, numValues);
//    prefix_sum_tree(x, numValues);
//    parallelPrefix(x,y,numValues);
    sequentialSum(x,numValues);
    std::cout << std::endl;
    free(x);
    free(y);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end-start;
    std::cout << duration.count() << std::endl;
    return 0;
}