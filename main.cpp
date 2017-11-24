#include <iostream>
#include <thread>
#include <ctime>
#include <cmath>

#define THREADS 4


void sequentialSum(int *array, int numElements) {
    for (int i = 1; i < numElements; ++i) {
        array[i] += array[i - 1];
    }
}

void scanNotWorkEfficient(int *array,int* read, int index, int iteration) {
    if (index >= pow(2,iteration - 1))
        array[index] = read[(int)(index - pow(2, iteration - 1))] + read[index];
}

int main() {
    const int numValues = 8;
    std::clock_t start;
    double duration;

    start = std::clock();

    int *x = (int *)malloc(sizeof(int) * numValues);
    int *y = (int *)malloc(sizeof(int) * numValues);

    for (int l = 0; l < numValues; ++l) {
        x[l] = l;
        y[l] = l;
    }
    int *aux;

    std::thread threads[numValues];
    const int iterations = (int)log2(numValues);
    for (int d = 1; d <= iterations; ++d) {
        threads[0] = std::thread(scanNotWorkEfficient, x, y, 0, d);
        for (int i = 1; i < numValues; ++i) {
            threads[i] = std::thread(scanNotWorkEfficient, x, y, i, d);
            if (!((i + 1) % THREADS))
                for (int j = THREADS - 1; j >= 0; --j) {
                    threads[i - j].join();
                }

        }
        std::cout << " ------------- x --------" << std::endl;

        for (int k = 0; k < 8; ++k) {
            std::cout << x[k] << " ";
        }
        std::cout << std::endl;
        std::cout << " ------------- y --------" << std::endl;
        for (int k = 0; k < 8; ++k) {
            std::cout << y[k] << " ";
        }
        std::cout << std::endl;
        aux = y;
        y = x;
        x = aux;
    }

    for (int k = 0; k < 8; ++k) {
        std::cout << y[k] << " ";
    }
    std::cout << std::endl;

    free(x);
    free(y);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
    return 0;
}