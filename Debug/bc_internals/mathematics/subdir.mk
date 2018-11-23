################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../bc_internals/mathematics/GPU.cu 

OBJS += \
./bc_internals/mathematics/GPU.o 

CU_DEPS += \
./bc_internals/mathematics/GPU.d 


# Each subdirectory must supply rules for building sources it contributes
bc_internals/mathematics/%.o: ../bc_internals/mathematics/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/usr/include/x86_64-linux-gnu -O0   -odir "bc_internals/mathematics" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/usr/include/x86_64-linux-gnu -O0 --compile --relocatable-device-code=false  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


