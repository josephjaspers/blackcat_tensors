################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../NN_Core/BlackCat_Tuple.cu \
../NN_Core/Defaults.cu \
../NN_Core/FeedForward.cu \
../NN_Core/FeedForwardInstantEval.cu \
../NN_Core/Layer.cu \
../NN_Core/OutputLayer.cu 

OBJS += \
./NN_Core/BlackCat_Tuple.o \
./NN_Core/Defaults.o \
./NN_Core/FeedForward.o \
./NN_Core/FeedForwardInstantEval.o \
./NN_Core/Layer.o \
./NN_Core/OutputLayer.o 

CU_DEPS += \
./NN_Core/BlackCat_Tuple.d \
./NN_Core/Defaults.d \
./NN_Core/FeedForward.d \
./NN_Core/FeedForwardInstantEval.d \
./NN_Core/Layer.d \
./NN_Core/OutputLayer.d 


# Each subdirectory must supply rules for building sources it contributes
NN_Core/%.o: ../NN_Core/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.3 -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -G -g -O0 -Xcompiler -fopenmp -gencode arch=compute_52,code=sm_52  -odir "NN_Core" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.3 -I/home/joseph/BlackCat_Libraries/BlackCat_Tensors3.2/BlackCat_Tensors_Functions/Functions -G -g -O0 -Xcompiler -fopenmp --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


