################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../BC_LAR_impl/LAR_transposeDeprecated/LAR_PointwiseTranspose_BoolVersion.cpp \
../BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeA.cpp \
../BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeAB.cpp \
../BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeB.cpp 

OBJS += \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_PointwiseTranspose_BoolVersion.o \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeA.o \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeAB.o \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeB.o 

CPP_DEPS += \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_PointwiseTranspose_BoolVersion.d \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeA.d \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeAB.d \
./BC_LAR_impl/LAR_transposeDeprecated/LAR_Pointwise_TransposeB.d 


# Each subdirectory must supply rules for building sources it contributes
BC_LAR_impl/LAR_transposeDeprecated/%.o: ../BC_LAR_impl/LAR_transposeDeprecated/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60  -odir "BC_LAR_impl/LAR_transposeDeprecated" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/home/joseph/cuda-workspace/BlackCat_Tensors/BC_Headers -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


