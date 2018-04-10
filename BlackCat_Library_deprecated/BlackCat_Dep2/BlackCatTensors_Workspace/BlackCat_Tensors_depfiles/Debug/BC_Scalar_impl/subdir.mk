################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Scalar_impl/Scalar_Constructors.cpp \
../BC_Scalar_impl/Scalar_Operators.cpp 

OBJS += \
./BC_Scalar_impl/Scalar_Constructors.o \
./BC_Scalar_impl/Scalar_Operators.o 

CPP_DEPS += \
./BC_Scalar_impl/Scalar_Constructors.d \
./BC_Scalar_impl/Scalar_Operators.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Scalar_impl/%.o: ../BC_Scalar_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "BC_Scalar_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


