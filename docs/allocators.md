# Allocators 

BlackCat Tensor's common-types (Vector, Matrix, Cube) are created using two template arguments. The first argument, is the scalar-type while the second is the allocator. This was designed to mimic `stl` library despite the fact that BCT's allocator's are NOT interchangeable to the standard libraries allocators. 

#### Differences between `std` and `BC` allocators 
1) BC allocators do not accept a template argument for the scalar_type. 
2) BC allocators are composed of entirely static methods. 
3) BC allocators define certain methods that are essential 'boiler plate' for Cuda. IE each allocator (including Basic_Allocator) define `HostToDevice(device_ptr, host_ptr, size)` and `DeviceToHost(host_ptr, device_ptr, size)`. These methods essential to copying to and from device memory. 

#### The primary allocators

	Basic_Allocator
	Cuda
	Cuda_Managed
	
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

#### Choosing an allocator
Example:
```
BC::Matrix<float> mat; 			    //defaults to Basic_Allocator
BC::Matrix<float, BC::Basic_Allocator> mat; //identical to above   
BC::Matrix<float, BC::Cuda> mat;	    //allocates memory on the GPU 
BC::Matrix<float, BC::Cuda_Managed> mat;    //allocates memory on the GPU but data transfer is managed automatically. 
```
