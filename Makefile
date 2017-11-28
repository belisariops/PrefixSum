
all:
	
	nvcc OpenCL/OpenCLPrefixSum.cpp Cuda/CudaPrefixSum.cpp Cuda/kernel.cu main.cpp -lOpenCL -o main
clean:
	rm main
