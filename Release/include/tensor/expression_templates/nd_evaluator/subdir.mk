################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../include/tensor/expression_templates/nd_evaluator/Device.cu \
../include/tensor/expression_templates/nd_evaluator/Device_Impl.cu 

OBJS += \
./include/tensor/expression_templates/nd_evaluator/Device.o \
./include/tensor/expression_templates/nd_evaluator/Device_Impl.o 

CU_DEPS += \
./include/tensor/expression_templates/nd_evaluator/Device.d \
./include/tensor/expression_templates/nd_evaluator/Device_Impl.d 


# Each subdirectory must supply rules for building sources it contributes
include/tensor/expression_templates/nd_evaluator/%.o: ../include/tensor/expression_templates/nd_evaluator/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_52,code=sm_52  -odir "include/tensor/expression_templates/nd_evaluator" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


