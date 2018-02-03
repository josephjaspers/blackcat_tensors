################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Expressions/Expression_Binary_Dotproduct.cu \
../BC_Expressions/Expression_Binary_Functors.cu \
../BC_Expressions/Expression_Binary_Pointwise_Same.cu \
../BC_Expressions/Expression_Binary_Pointwise_Scalar.cu \
../BC_Expressions/Expression_Unary_MatrixTransposition.cu \
../BC_Expressions/SubTensor_Expression.cu 

OBJS += \
./BC_Expressions/Expression_Binary_Dotproduct.o \
./BC_Expressions/Expression_Binary_Functors.o \
./BC_Expressions/Expression_Binary_Pointwise_Same.o \
./BC_Expressions/Expression_Binary_Pointwise_Scalar.o \
./BC_Expressions/Expression_Unary_MatrixTransposition.o \
./BC_Expressions/SubTensor_Expression.o 

CU_DEPS += \
./BC_Expressions/Expression_Binary_Dotproduct.d \
./BC_Expressions/Expression_Binary_Functors.d \
./BC_Expressions/Expression_Binary_Pointwise_Same.d \
./BC_Expressions/Expression_Binary_Pointwise_Scalar.d \
./BC_Expressions/Expression_Unary_MatrixTransposition.d \
./BC_Expressions/SubTensor_Expression.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Expressions/%.o: ../BC_Expressions/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "BC_Expressions" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


