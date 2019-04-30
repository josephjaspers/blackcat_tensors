################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../benchmarks/coefficientwise/cu_cwise.cu 

OBJS += \
./benchmarks/coefficientwise/cu_cwise.o 

CU_DEPS += \
./benchmarks/coefficientwise/cu_cwise.d 


# Each subdirectory must supply rules for building sources it contributes
benchmarks/coefficientwise/%.o: ../benchmarks/coefficientwise/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_52,code=sm_52  -odir "benchmarks/coefficientwise" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


