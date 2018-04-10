################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_LAR_impl/LAR_DotProducts.cpp \
../BC_LAR_impl/LAR_General.cpp \
../BC_LAR_impl/LAR_Indexing.cpp \
../BC_LAR_impl/LAR_Pointwise.cpp \
../BC_LAR_impl/LAR_PointwiseHighDegree.cpp \
../BC_LAR_impl/LAR_Pointwise_Specialized.cpp \
../BC_LAR_impl/LAR_advanced.cpp 

OBJS += \
./BC_LAR_impl/LAR_DotProducts.o \
./BC_LAR_impl/LAR_General.o \
./BC_LAR_impl/LAR_Indexing.o \
./BC_LAR_impl/LAR_Pointwise.o \
./BC_LAR_impl/LAR_PointwiseHighDegree.o \
./BC_LAR_impl/LAR_Pointwise_Specialized.o \
./BC_LAR_impl/LAR_advanced.o 

CPP_DEPS += \
./BC_LAR_impl/LAR_DotProducts.d \
./BC_LAR_impl/LAR_General.d \
./BC_LAR_impl/LAR_Indexing.d \
./BC_LAR_impl/LAR_Pointwise.d \
./BC_LAR_impl/LAR_PointwiseHighDegree.d \
./BC_LAR_impl/LAR_Pointwise_Specialized.d \
./BC_LAR_impl/LAR_advanced.d 


# Each subdirectory must supply rules for building sources it contributes
BC_LAR_impl/%.o: ../BC_LAR_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "BC_LAR_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


