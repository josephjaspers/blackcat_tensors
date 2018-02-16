################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../BC_Internals/BC_Expressions/Expression_Base.cu \
../BC_Internals/BC_Expressions/Expression_Binary_Dotproduct.cu \
../BC_Internals/BC_Expressions/Expression_Binary_Dotproduct_impl.cu \
../BC_Internals/BC_Expressions/Expression_Binary_Functors.cu \
../BC_Internals/BC_Expressions/Expression_Binary_Pointwise.cu \
../BC_Internals/BC_Expressions/Expression_Binary_Pointwise_Scalar.cu \
../BC_Internals/BC_Expressions/Expression_Unary_MatrixTransposition.cu \
../BC_Internals/BC_Expressions/Expression_Unary_Negation.cu \
../BC_Internals/BC_Expressions/Expression_Unary_Pointwise.cu 

OBJS += \
./BC_Internals/BC_Expressions/Expression_Base.o \
./BC_Internals/BC_Expressions/Expression_Binary_Dotproduct.o \
./BC_Internals/BC_Expressions/Expression_Binary_Dotproduct_impl.o \
./BC_Internals/BC_Expressions/Expression_Binary_Functors.o \
./BC_Internals/BC_Expressions/Expression_Binary_Pointwise.o \
./BC_Internals/BC_Expressions/Expression_Binary_Pointwise_Scalar.o \
./BC_Internals/BC_Expressions/Expression_Unary_MatrixTransposition.o \
./BC_Internals/BC_Expressions/Expression_Unary_Negation.o \
./BC_Internals/BC_Expressions/Expression_Unary_Pointwise.o 

CU_DEPS += \
./BC_Internals/BC_Expressions/Expression_Base.d \
./BC_Internals/BC_Expressions/Expression_Binary_Dotproduct.d \
./BC_Internals/BC_Expressions/Expression_Binary_Dotproduct_impl.d \
./BC_Internals/BC_Expressions/Expression_Binary_Functors.d \
./BC_Internals/BC_Expressions/Expression_Binary_Pointwise.d \
./BC_Internals/BC_Expressions/Expression_Binary_Pointwise_Scalar.d \
./BC_Internals/BC_Expressions/Expression_Unary_MatrixTransposition.d \
./BC_Internals/BC_Expressions/Expression_Unary_Negation.d \
./BC_Internals/BC_Expressions/Expression_Unary_Pointwise.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Internals/BC_Expressions/%.o: ../BC_Internals/BC_Expressions/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -O3 -Xcompiler -fopenmp -gencode arch=compute_52,code=sm_52  -odir "BC_Internals/BC_Expressions" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -O3 -Xcompiler -fopenmp --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


