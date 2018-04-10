################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_LAR_impl/LAR_DotProducts.cpp \
../BC_LAR_impl/LAR_General.cpp \
../BC_LAR_impl/LAR_Indexing.cpp \
../BC_LAR_impl/LAR_Pointwise.cpp \
../BC_LAR_impl/LAR_Pointwise_Specialized.cpp \
../BC_LAR_impl/LAR_advanced.cpp 

OBJS += \
./BC_LAR_impl/LAR_DotProducts.o \
./BC_LAR_impl/LAR_General.o \
./BC_LAR_impl/LAR_Indexing.o \
./BC_LAR_impl/LAR_Pointwise.o \
./BC_LAR_impl/LAR_Pointwise_Specialized.o \
./BC_LAR_impl/LAR_advanced.o 

CPP_DEPS += \
./BC_LAR_impl/LAR_DotProducts.d \
./BC_LAR_impl/LAR_General.d \
./BC_LAR_impl/LAR_Indexing.d \
./BC_LAR_impl/LAR_Pointwise.d \
./BC_LAR_impl/LAR_Pointwise_Specialized.d \
./BC_LAR_impl/LAR_advanced.d 


# Each subdirectory must supply rules for building sources it contributes
BC_LAR_impl/%.o: ../BC_LAR_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2 -std=c++11 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "BC_LAR_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


