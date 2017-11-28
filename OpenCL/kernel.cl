kernel void naiveSum( global int* A, global int* B , global int size) {
    const int index = get_global_id(0);
}

kernel void upSweep( global int* A, global int* B , global int size) {
    const int index = get_global_id(0);
}

kernel void downSweep( global int* A, global int* B , global int size) {
    const int index = get_global_id(0);
}