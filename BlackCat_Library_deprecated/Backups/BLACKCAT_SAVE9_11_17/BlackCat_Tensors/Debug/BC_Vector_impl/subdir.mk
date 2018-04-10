################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_Vector_impl/Vector_AssignmentOperators.cpp \
../BC_Vector_impl/Vector_BoundryChecking.cpp \
../BC_Vector_impl/Vector_Constructors.cpp \
../BC_Vector_impl/Vector_PointwiseOperators.cpp 

OBJS += \
./BC_Vector_impl/Vector_AssignmentOperators.o \
./BC_Vector_impl/Vector_BoundryChecking.o \
./BC_Vector_impl/Vector_Constructors.o \
./BC_Vector_impl/Vector_PointwiseOperators.o 

CPP_DEPS += \
./BC_Vector_impl/Vector_AssignmentOperators.d \
./BC_Vector_impl/Vector_BoundryChecking.d \
./BC_Vector_impl/Vector_Constructors.d \
./BC_Vector_impl/Vector_PointwiseOperators.d 


# Each subdirectory must supply rules for building sources it contributes
BC_Vector_impl/%.o: ../BC_Vector_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_Vector_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


