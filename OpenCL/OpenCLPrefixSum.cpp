//
// Created by belisariops on 11/27/17.
//

#include <iostream>
#include "OpenCLPrefixSum.h"

OpenCLPrefixSum::OpenCLPrefixSum(int size) {
    N_ELEMENTS= size;
    int platform_id=0, device_id=0;

    try {
        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices;
        // Select the platform.
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

        // Create a context
        cl::Context context(devices);

        // Create a command queue
        // Select the device.
        queue = cl::CommandQueue(context, devices[device_id]);

        // Create the memory buffers
        bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
        bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, N_ELEMENTS* sizeof(int));

        // Read the program source
        std::ifstream sourceFile("OpenCL/kernel.cl");
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));

        // Make program from the source code
        cl::Program program = cl::Program(context, source);

        // Build the program for the devices
        program.build(devices);

        // Make kernel
        kernel_1 = cl::Kernel(program, "naiveSum");
        kernel_2 = cl::Kernel(program, "upSweep");
        kernel_3 = cl::Kernel(program, "downSweep");
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

}

void OpenCLPrefixSum::runNaiveSum(int *A, int size) {
    try {
        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, N_ELEMENTS * sizeof(int), A);
        queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, N_ELEMENTS * sizeof(int), A);

        // Set the kernel arguments
        kernel_1.setArg( 0, bufferA );
        kernel_1.setArg( 1, bufferB );
        kernel_1.setArg( 2, size );
        kernel_1.setArg( 3, 1 );

        // Execute the kernel
        cl::NDRange global( N_ELEMENTS );
        cl::NDRange local( 1 );
        queue.enqueueNDRangeKernel( kernel_1, cl::NullRange, global, local );

        // Copy the output data back to the host
        queue.enqueueReadBuffer( bufferB, CL_TRUE, 0, N_ELEMENTS * sizeof(int), A );
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

}

OpenCLPrefixSum::~OpenCLPrefixSum() {

}

void OpenCLPrefixSum::runPrefixSum(int *A, int size) {

    try {
        // Copy the input data to the input buffers using the command queue.
        queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, N_ELEMENTS * sizeof(int), A);
        // Set the kernel arguments
        kernel_2.setArg( 0, bufferA );
        kernel_2.setArg( 1, size );

        // Execute the kernel
        cl::NDRange global( N_ELEMENTS );
        cl::NDRange local( 1 );
        queue.enqueueNDRangeKernel( kernel_2, cl::NullRange, global, local );

        // Set the kernel arguments
        kernel_3.setArg( 0, bufferA );
        kernel_3.setArg( 1, size );

        // Execute the kernel
        queue.enqueueNDRangeKernel( kernel_3, cl::NullRange, global, local );


        // Copy the output data back to the host
        queue.enqueueReadBuffer( bufferA, CL_TRUE, 0, N_ELEMENTS * sizeof(int), A );
    }
    catch (cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
    }

}
