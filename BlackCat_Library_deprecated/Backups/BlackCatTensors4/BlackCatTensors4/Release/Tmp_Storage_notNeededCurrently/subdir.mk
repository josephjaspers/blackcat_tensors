################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Tmp_Storage_notNeededCurrently/armadillo_test.cpp 

OBJS += \
./Tmp_Storage_notNeededCurrently/armadillo_test.o 

CPP_DEPS += \
./Tmp_Storage_notNeededCurrently/armadillo_test.d 


# Each subdirectory must supply rules for building sources it contributes
Tmp_Storage_notNeededCurrently/%.o: ../Tmp_Storage_notNeededCurrently/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -fopenmp -ftemplate-depth=10005 -std=c++17 -O3  -larmadillo  -llapack -lblas -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


