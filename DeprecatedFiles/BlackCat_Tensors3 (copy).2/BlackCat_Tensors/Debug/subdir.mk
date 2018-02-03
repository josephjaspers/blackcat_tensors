################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../NN_test.cu \
../SpeedTests.cu \
../main.cu \
../test_area.cu 

OBJS += \
./NN_test.o \
./SpeedTests.o \
./main.o \
./test_area.o 

CU_DEPS += \
./NN_test.d \
./SpeedTests.d \
./main.d \
./test_area.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -G -g -O0 -gencode arch=compute_52,code=sm_52  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/usr/local/cuda-9.1/bin -I/usr/local/cuda/include -I/usr/local/include/atlas-base -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


