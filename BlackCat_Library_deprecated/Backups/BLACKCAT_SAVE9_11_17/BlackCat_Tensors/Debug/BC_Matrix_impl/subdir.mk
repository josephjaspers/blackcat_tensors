################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Matrix_impl/Matrix_BasicAccessors.cpp 

OBJS += \
./BC_Matrix_impl/Matrix_BasicAccessors.o 

CPP_DEPS += \
./BC_Matrix_impl/Matrix_BasicAccessors.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Matrix_impl/%.o: ../BC_Matrix_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_Matrix_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


