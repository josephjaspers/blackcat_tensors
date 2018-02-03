################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/BC_MathLibraries/Mathematics_GPU.cu \
../BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.cu \
../BC_Internals/BC_MathLibraries/Mathematics_Opt.cu \
../BC_Internals/BC_MathLibraries/Mathematics_mGPU.cu 

OBJS += \
./BC_Internals/BC_MathLibraries/Mathematics_GPU.o \
./BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.o \
./BC_Internals/BC_MathLibraries/Mathematics_Opt.o \
./BC_Internals/BC_MathLibraries/Mathematics_mGPU.o 

CU_DEPS += \
./BC_Internals/BC_MathLibraries/Mathematics_GPU.d \
./BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.d \
./BC_Internals/BC_MathLibraries/Mathematics_Opt.d \
./BC_Internals/BC_MathLibraries/Mathematics_mGPU.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/BC_MathLibraries/%.o: ../BC_Internals/BC_MathLibraries/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "BC_Internals/BC_MathLibraries" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


