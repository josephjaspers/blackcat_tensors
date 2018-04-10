################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../NN/MNIST_test.cpp 

OBJS += \
./NN/MNIST_test.o 

CPP_DEPS += \
./NN/MNIST_test.d 


# Each subdirectory must supply rules for building sources it contributes
NN/%.o: ../NN/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++17 -w -fopenmp -I/usr/lib/atlas-base  -I/usr/include/atlas-base -lcblas -lf77blas -I/usr/include/atlas -I/home/joseph/BlackCatTensor_NsightWorkSpace/BlackCat_Tensors/Debug -I/home/joseph/BlackCatTensor_NsightWorkSpace/BlackCat_Tensors/BC_Headers -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


