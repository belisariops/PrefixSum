kernel void naiveSum( global int* A, global int* B , int size, int iteration) {
    const int index = get_global_id(0);
    if (index >= (1 << (iteration - 1)))
        A[index] = B[(int) (index - (1 << (iteration - 1)))] + B[index];
    else
        A[index] = B[index];
}

kernel void upSweep( global int* A, int size, int iteration) {
    const int index = get_global_id(0);
    if (!((index + 1) % (1 << (iteration + 1))))
        A[index] = A[index - (1<<iteration)] + A[index];
}

kernel void downSweep( global int* A, int size, int iteration) {
    const int index = get_global_id(0);
    int aux;
    if (!((index + 1) % (1 << (iteration + 1)))) {
        aux = A[index - (1<<iteration)];
        A[index - (1<<iteration)] = A[index];
        A[index] = aux + A[index];
    }
}

kernel void setLastToCero(global int* A, int size) {
    const int index = get_global_id(0);
    if (index == size - 1)
        A[index] = 0;
}