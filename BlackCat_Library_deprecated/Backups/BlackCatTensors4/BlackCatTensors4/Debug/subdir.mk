################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../armadillo_test.cpp \
../lazy_inst_test.cpp \
../test.cpp 

OBJS += \
./armadillo_test.o \
./lazy_inst_test.o \
./test.o 

CPP_DEPS += \
./armadillo_test.d \
./lazy_inst_test.d \
./test.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -fopenmp -ftemplate-depth=1000000 -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


