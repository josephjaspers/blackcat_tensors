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
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.3 -O3 -gencode arch=compute_52,code=sm_52  -odir "UnitTests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.3 -O3 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


