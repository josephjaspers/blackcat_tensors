################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Headers/BC_Matrix_impl/Matrix_BasicAccessors.cpp 

OBJS += \
./BC_Headers/BC_Matrix_impl/Matrix_BasicAccessors.o 

CPP_DEPS += \
./BC_Headers/BC_Matrix_impl/Matrix_BasicAccessors.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Headers/BC_Matrix_impl/%.o: ../BC_Headers/BC_Matrix_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0   -odir "BC_Headers/BC_Matrix_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


