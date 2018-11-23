################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation/GPU_impl.cu 

OBJS += \
./BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation/GPU_impl.o 

CU_DEPS += \
./BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation/GPU_impl.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation/%.o: ../BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -I/usr/include/x86_64-linux-gnu -G -g -O0   -odir "BC_Internals/BC_Tensor/BC_MathLibraries/GPU_Implementation" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -I/usr/include/x86_64-linux-gnu -G -g -O0 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


