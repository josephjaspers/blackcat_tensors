################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../UnitTests/MNIST_test.cu \
../UnitTests/MNIST_test_WithMultiThreading.cu \
../UnitTests/MNIST_test_WithReadWriteExample.cu \
../UnitTests/XOR_test.cu 

OBJS += \
./UnitTests/MNIST_test.o \
./UnitTests/MNIST_test_WithMultiThreading.o \
./UnitTests/MNIST_test_WithReadWriteExample.o \
./UnitTests/XOR_test.o 

CU_DEPS += \
./UnitTests/MNIST_test.d \
./UnitTests/MNIST_test_WithMultiThreading.d \
./UnitTests/MNIST_test_WithReadWriteExample.d \
./UnitTests/XOR_test.d 


# Each subdirectory must supply rules for building sources it contributes
UnitTests/%.o: ../UnitTests/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -O3 -gencode arch=compute_52,code=sm_52  -odir "UnitTests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -O3 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


