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
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0   -odir "BC_LAR_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


