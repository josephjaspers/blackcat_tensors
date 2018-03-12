
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

static const int WORK_SIZE = 256;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

int main(int argc, char **argv)
{
	CUmodule module;
	CUcontext context;
	CUdevice device;
	CUdeviceptr deviceArray;
	CUfunction process;

	void *kernelArguments[] = { &deviceArray };
	int deviceCount;
	unsigned int idata[WORK_SIZE], odata[WORK_SIZE];

	for (int i = 0; i < WORK_SIZE; ++i) {
		idata[i] = i;
	}

	CHECK_CUDA_RESULT(cuInit(0));
	CHECK_CUDA_RESULT(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) {
		printf("No CUDA-compatible devices found\n");
		exit(1);
	}
	CHECK_CUDA_RESULT(cuDeviceGet(&device, 0));
	CHECK_CUDA_RESULT(cuCtxCreate(&context, 0, device));

	CHECK_CUDA_RESULT(cuModuleLoad(&module, "bitreverse.fatbin"));
	CHECK_CUDA_RESULT(cuModuleGetFunction(&process, module, "bitreverse"));

	CHECK_CUDA_RESULT(cuMemAlloc(&deviceArray, sizeof(int) * WORK_SIZE));
	CHECK_CUDA_RESULT(
			cuMemcpyHtoD(deviceArray, idata, sizeof(int) * WORK_SIZE));

	CHECK_CUDA_RESULT(
			cuLaunchKernel(process, 1, 1, 1, WORK_SIZE, 1, 1, 0, NULL, kernelArguments, NULL));

	CHECK_CUDA_RESULT(
			cuMemcpyDtoH(odata, deviceArray, sizeof(int) * WORK_SIZE));

	for (int i = 0; i < WORK_SIZE; ++i) {
		printf("Input value: %u, output value: %u\n", idata[i], odata[i]);
	}

	CHECK_CUDA_RESULT(cuMemFree(deviceArray));
	CHECK_CUDA_RESULT(cuCtxDestroy(context));

	return 0;
}
