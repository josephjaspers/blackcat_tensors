################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../UnitTests/main.cu 

OBJS += \
./UnitTests/main.o 

CU_DEPS += \
./UnitTests/main.d 


# Each subdirectory must supply rules for building sources it contributes
UnitTests/%.o: ../UnitTests/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "UnitTests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


