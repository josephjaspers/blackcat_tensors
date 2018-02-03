################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_MathLibraries/Mathematics_GPU.cu \
../BC_MathLibraries/Mathematics_GPU_impl.cu \
../BC_MathLibraries/Mathematics_Opt.cu 

OBJS += \
./BC_MathLibraries/Mathematics_GPU.o \
./BC_MathLibraries/Mathematics_GPU_impl.o \
./BC_MathLibraries/Mathematics_Opt.o 

CU_DEPS += \
./BC_MathLibraries/Mathematics_GPU.d \
./BC_MathLibraries/Mathematics_GPU_impl.d \
./BC_MathLibraries/Mathematics_Opt.d 


# Each subdirectory must supply rules for building sources it contributes
BC_MathLibraries/%.o: ../BC_MathLibraries/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "BC_MathLibraries" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


