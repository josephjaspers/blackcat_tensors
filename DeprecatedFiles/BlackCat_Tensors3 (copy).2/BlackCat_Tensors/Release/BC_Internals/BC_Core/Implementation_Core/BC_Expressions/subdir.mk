################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Dotproduct.cu \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Functors.cu \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Same.cu \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Scalar.cu \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Unary_MatrixTransposition.cu \
../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/SubTensor_Expression.cu 

OBJS += \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Dotproduct.o \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Functors.o \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Same.o \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Scalar.o \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Unary_MatrixTransposition.o \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/SubTensor_Expression.o 

CU_DEPS += \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Dotproduct.d \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Functors.d \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Same.d \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Binary_Pointwise_Scalar.d \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/Expression_Unary_MatrixTransposition.d \
./BC_Internals/BC_Core/Implementation_Core/BC_Expressions/SubTensor_Expression.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/BC_Core/Implementation_Core/BC_Expressions/%.o: ../BC_Internals/BC_Core/Implementation_Core/BC_Expressions/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -O3 -gencode arch=compute_52,code=sm_52  -odir "BC_Internals/BC_Core/Implementation_Core/BC_Expressions" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -O3 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


