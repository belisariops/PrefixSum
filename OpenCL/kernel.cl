kernel void naiveSum( global int* A, global int* B , global int size, global int iteration) {
    const int index = get_global_id(0);
    if (index >= (1 << (iteration - 1)))
        A[index] = B[(int) (index - (1 << (iteration - 1)))] + B[index];
    else
        A[index] = B[index];
}

kernel void upSweep( global int* A, global int* B , global int size) {
    const int index = get_global_id(0);
}

kernel void downSweep( global int* A, global int* B , global int size) {
    const int index = get_global_id(0);
}