################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Tensor_impl/Tensor_AdvancedOperations.cpp \
../BC_Tensor_impl/Tensor_AssignmentOperators.cpp \
../BC_Tensor_impl/Tensor_BasicAccessors.cpp \
../BC_Tensor_impl/Tensor_BoundryChecks.cpp \
../BC_Tensor_impl/Tensor_Constructors.cpp \
../BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.cpp \
../BC_Tensor_impl/Tensor_DataAccessors.cpp \
../BC_Tensor_impl/Tensor_General.cpp \
../BC_Tensor_impl/Tensor_MovementSemantics.cpp \
../BC_Tensor_impl/Tensor_PointwiseOperators.cpp 

OBJS += \
./BC_Tensor_impl/Tensor_AdvancedOperations.o \
./BC_Tensor_impl/Tensor_AssignmentOperators.o \
./BC_Tensor_impl/Tensor_BasicAccessors.o \
./BC_Tensor_impl/Tensor_BoundryChecks.o \
./BC_Tensor_impl/Tensor_Constructors.o \
./BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.o \
./BC_Tensor_impl/Tensor_DataAccessors.o \
./BC_Tensor_impl/Tensor_General.o \
./BC_Tensor_impl/Tensor_MovementSemantics.o \
./BC_Tensor_impl/Tensor_PointwiseOperators.o 

CPP_DEPS += \
./BC_Tensor_impl/Tensor_AdvancedOperations.d \
./BC_Tensor_impl/Tensor_AssignmentOperators.d \
./BC_Tensor_impl/Tensor_BasicAccessors.d \
./BC_Tensor_impl/Tensor_BoundryChecks.d \
./BC_Tensor_impl/Tensor_Constructors.d \
./BC_Tensor_impl/Tensor_Constructors_PrivateDotProducts.d \
./BC_Tensor_impl/Tensor_DataAccessors.d \
./BC_Tensor_impl/Tensor_General.d \
./BC_Tensor_impl/Tensor_MovementSemantics.d \
./BC_Tensor_impl/Tensor_PointwiseOperators.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Tensor_impl/%.o: ../BC_Tensor_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_Tensor_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


