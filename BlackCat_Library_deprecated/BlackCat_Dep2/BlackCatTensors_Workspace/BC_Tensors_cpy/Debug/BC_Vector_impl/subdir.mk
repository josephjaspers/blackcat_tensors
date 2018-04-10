################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Vector_impl/Vector_AssignmentOperators.cpp \
../BC_Vector_impl/Vector_BasicAccessors.cpp \
../BC_Vector_impl/Vector_BoundryChecking.cpp \
../BC_Vector_impl/Vector_Constructors.cpp \
../BC_Vector_impl/Vector_PointwiseOperators.cpp 

OBJS += \
./BC_Vector_impl/Vector_AssignmentOperators.o \
./BC_Vector_impl/Vector_BasicAccessors.o \
./BC_Vector_impl/Vector_BoundryChecking.o \
./BC_Vector_impl/Vector_Constructors.o \
./BC_Vector_impl/Vector_PointwiseOperators.o 

CPP_DEPS += \
./BC_Vector_impl/Vector_AssignmentOperators.d \
./BC_Vector_impl/Vector_BasicAccessors.d \
./BC_Vector_impl/Vector_BoundryChecking.d \
./BC_Vector_impl/Vector_Constructors.d \
./BC_Vector_impl/Vector_PointwiseOperators.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Vector_impl/%.o: ../BC_Vector_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0   -odir "BC_Vector_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


