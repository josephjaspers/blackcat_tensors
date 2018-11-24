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

BlackCat Tensors attempts to align itself with the design of the C++ standard library to facilitate predictable syntax, and integration with the standard template library. Its design and documentation is divided amongst module-like components that are frequently concepts of the standard library. 

|Index| Concept | Brief |
| --- | --- | --- |
| 1 | [Core Types](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/aliases.md) | List of primary Tensor-Types |
| 2 | [Allocators](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/allocators.md) | Overview of allocators in BCT and their functionality |
| 3 | [Iterators](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/iterators.md) | Overview of iterators in BCT and their functionality |
| 4 | [Algorithms](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/algorithms.md) | Brief overview of how algorithms are implemented, and a list of supported algorithms |
| 5 | [CMath Functions](https://github.com/josephjaspers/BlackCat_Tensors/blob/master/docs/cmath_functions.md) | A list of supported CMath functions |
