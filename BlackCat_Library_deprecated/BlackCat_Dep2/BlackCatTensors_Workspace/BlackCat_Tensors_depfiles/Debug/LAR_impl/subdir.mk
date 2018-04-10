################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../LAR_impl/LAR_DotProducts.cpp \
../LAR_impl/LAR_General.cpp \
../LAR_impl/LAR_Indexing.cpp \
../LAR_impl/LAR_Pointwise.cpp \
../LAR_impl/LAR_PointwiseTranspose_BoolVersion.cpp \
../LAR_impl/LAR_Pointwise_TransposeA.cpp \
../LAR_impl/LAR_Pointwise_TransposeAB.cpp \
../LAR_impl/LAR_Pointwise_TransposeB.cpp \
../LAR_impl/LAR_advanced.cpp 

OBJS += \
./LAR_impl/LAR_DotProducts.o \
./LAR_impl/LAR_General.o \
./LAR_impl/LAR_Indexing.o \
./LAR_impl/LAR_Pointwise.o \
./LAR_impl/LAR_PointwiseTranspose_BoolVersion.o \
./LAR_impl/LAR_Pointwise_TransposeA.o \
./LAR_impl/LAR_Pointwise_TransposeAB.o \
./LAR_impl/LAR_Pointwise_TransposeB.o \
./LAR_impl/LAR_advanced.o 

CPP_DEPS += \
./LAR_impl/LAR_DotProducts.d \
./LAR_impl/LAR_General.d \
./LAR_impl/LAR_Indexing.d \
./LAR_impl/LAR_Pointwise.d \
./LAR_impl/LAR_PointwiseTranspose_BoolVersion.d \
./LAR_impl/LAR_Pointwise_TransposeA.d \
./LAR_impl/LAR_Pointwise_TransposeAB.d \
./LAR_impl/LAR_Pointwise_TransposeB.d \
./LAR_impl/LAR_advanced.d 


# Each subdirectory must supply rules for building sources it contributes
LAR_impl/%.o: ../LAR_impl/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "LAR_impl" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


