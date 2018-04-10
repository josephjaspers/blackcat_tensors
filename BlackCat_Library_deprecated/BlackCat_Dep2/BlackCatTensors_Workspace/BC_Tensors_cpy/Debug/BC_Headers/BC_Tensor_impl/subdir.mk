################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Headers/BC_Tensor_impl/Tensor_AssignmentOperators.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_BasicAccessors.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_Constructors.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_DataAccessors.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_DotProducts.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_General.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_MiscMethods.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_MovementSemantics.cpp \
../BC_Headers/BC_Tensor_impl/Tensor_PointwiseOperators.cpp 

OBJS += \
./BC_Headers/BC_Tensor_impl/Tensor_AssignmentOperators.o \
./BC_Headers/BC_Tensor_impl/Tensor_BasicAccessors.o \
./BC_Headers/BC_Tensor_impl/Tensor_Constructors.o \
./BC_Headers/BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.o \
./BC_Headers/BC_Tensor_impl/Tensor_DataAccessors.o \
./BC_Headers/BC_Tensor_impl/Tensor_DotProducts.o \
./BC_Headers/BC_Tensor_impl/Tensor_General.o \
./BC_Headers/BC_Tensor_impl/Tensor_MiscMethods.o \
./BC_Headers/BC_Tensor_impl/Tensor_MovementSemantics.o \
./BC_Headers/BC_Tensor_impl/Tensor_PointwiseOperators.o 

CPP_DEPS += \
./BC_Headers/BC_Tensor_impl/Tensor_AssignmentOperators.d \
./BC_Headers/BC_Tensor_impl/Tensor_BasicAccessors.d \
./BC_Headers/BC_Tensor_impl/Tensor_Constructors.d \
./BC_Headers/BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.d \
./BC_Headers/BC_Tensor_impl/Tensor_DataAccessors.d \
./BC_Headers/BC_Tensor_impl/Tensor_DotProducts.d \
./BC_Headers/BC_Tensor_impl/Tensor_General.d \
./BC_Headers/BC_Tensor_impl/Tensor_MiscMethods.d \
./BC_Headers/BC_Tensor_impl/Tensor_MovementSemantics.d \
./BC_Headers/BC_Tensor_impl/Tensor_PointwiseOperators.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Headers/BC_Tensor_impl/%.o: ../BC_Headers/BC_Tensor_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0   -odir "BC_Headers/BC_Tensor_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


