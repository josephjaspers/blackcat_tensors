################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BC_Tensors_test.cpp 

OBJS += \
./src/BC_Tensors_test.o 

CPP_DEPS += \
./src/BC_Tensors_test.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/Documents/CBlas_Libraries -O3 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -I/home/joseph/Documents/CBlas_Libraries -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


