################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/BC_MathLibraries/Mathematics_GPU.cu \
../BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.cu 

OBJS += \
./BC_Internals/BC_MathLibraries/Mathematics_GPU.o \
./BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.o 

CU_DEPS += \
./BC_Internals/BC_MathLibraries/Mathematics_GPU.d \
./BC_Internals/BC_MathLibraries/Mathematics_GPU_impl.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/BC_MathLibraries/%.o: ../BC_Internals/BC_MathLibraries/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0   -odir "BC_Internals/BC_MathLibraries" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


