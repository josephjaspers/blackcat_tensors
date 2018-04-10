################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../LazyInst.cpp \
../main3.cpp 

OBJS += \
./LazyInst.o \
./main3.o 

CPP_DEPS += \
./LazyInst.d \
./main3.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++17 -w -O2 -ftemplate-depth-10001 -I/usr/include/atlas -I/home/joseph/BlackCatTensor_NsightWorkSpace/BlackCat_Tensors/Debug -I/home/joseph/BlackCatTensor_NsightWorkSpace/BlackCat_Tensors/BC_Headers -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


