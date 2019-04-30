################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../include/blas/device_unsupported/GPU_Convolution_impl.cu 

OBJS += \
./include/blas/device_unsupported/GPU_Convolution_impl.o 

CU_DEPS += \
./include/blas/device_unsupported/GPU_Convolution_impl.d 


# Each subdirectory must supply rules for building sources it contributes
include/blas/device_unsupported/%.o: ../include/blas/device_unsupported/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_52,code=sm_52  -odir "include/blas/device_unsupported" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


