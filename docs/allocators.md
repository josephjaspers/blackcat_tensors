# Allocators 

BlackCat Tensor's common-types (Vector, Matrix, Cube) are created using two template arguments. The first argument, is the scalar-type while the second is the allocator. This was designed to mimic `stl` library.

#### Note: Currently allocators must be trivially copyable. (This will be changed in the future).

	
#### Overview

*Basic_Allocator*:

	The standard allocator and default argument.
	The Basic Allocator is comparable to the std::allocator.
	It has no special properties.

*Cuda*:

	The generic allocator for CUDA. 
	Allocates memory on the device. 
	Memory allocated with Cuda cannot be accessed from the host.

*Cuda_Managed*:

	Cuda_Managed allocates memory via CudaMallocManaged. 
	The memory is accessible from both host and device. 
	Cuda_Managed is recommended if you are not familiar with Cuda programming.

#### Choosing an allocator (example):

```cpp
BC::Matrix<float> mat; 			    	   //defaults to Basic_Allocator<float>
BC::Matrix<float, BC::Basic_Allocator<float>> mat; //identical to above   
BC::Matrix<float, BC::Cuda<float>> mat;	    	   //allocates memory on the GPU 
BC::Matrix<float, BC::Cuda_Managed<float>> mat;    //allocates memory on the GPU but data transfer is managed automatically. 
```
