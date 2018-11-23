################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/mathematics/GPU_Implementation/Unsupported/GPU_Convolution_impl.cu 

OBJS += \
./BC_Internals/mathematics/GPU_Implementation/Unsupported/GPU_Convolution_impl.o 

CU_DEPS += \
./BC_Internals/mathematics/GPU_Implementation/Unsupported/GPU_Convolution_impl.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/mathematics/GPU_Implementation/Unsupported/%.o: ../BC_Internals/mathematics/GPU_Implementation/Unsupported/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -I/usr/include/x86_64-linux-gnu -O0   -odir "BC_Internals/mathematics/GPU_Implementation/Unsupported" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -I/usr/include/x86_64-linux-gnu -O0 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


