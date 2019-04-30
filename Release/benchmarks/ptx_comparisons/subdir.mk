################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../benchmarks/ptx_comparisons/ptx_comparison_1.cu 

OBJS += \
./benchmarks/ptx_comparisons/ptx_comparison_1.o 

CU_DEPS += \
./benchmarks/ptx_comparisons/ptx_comparison_1.d 


# Each subdirectory must supply rules for building sources it contributes
benchmarks/ptx_comparisons/%.o: ../benchmarks/ptx_comparisons/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_52,code=sm_52  -odir "benchmarks/ptx_comparisons" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


