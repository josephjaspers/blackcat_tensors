# Documentation 
Author: Joseph Jaspers

BlackCat_Tensors (BCT) is a highly optimized Matrix library designed for NeuralNetwork construction. BCT is designed to support GPU computing (CUDA) and CPU multi-threading (OpenMP).BCT focuses on delivering a high-level framework for Neural Network construction with low-level performance. 

#### Intallation/Setup:
BCT is a header only library that supports compilation with the NVCC and G++ BCT does not support any default BLAS routines, and must be linked with an apropriate BLAS library. Setting up simply requires adding the BlackCat_Tensors your path and including "BlackCat_Tensors.h"

#### FAQ Fast Explanation:

CPU multithreading? Simply link OpenMP
GPU multithreading? Simply link CUDA 9

How to choose allocation?

```cpp
BC::Vector<float, BC::Cuda> myVec(sz);             //Allocates data on the gpu
BC::Vector<double, BC::Basic_Allocator> myVec(sz); //Allocates data on the cpu
BC::Vector<double>  myVec(sz);                     //defaults to BC::Basic_Allocator
```
**Must be linked to an apropriate BLAS with cblas_dgemm function and cblas_sgemm function.
**Dotproduct currently only available to double, and float types.
**CUDA BLAS routines only support floats. 

Non-numeric types are supported, though non-numeric types are not heavily tested in release. 

#### Concept Index  
1. [Core Types](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/aliases.md)
2. [Allocators](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/allocators.md)
3. [Iterators](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/iterators.md)
4. [Expression Templates](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/expression_templates.md)
5. [Algorithms](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/algorithms.md)
6. [CMath Functions](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/cmath_functions.md)
