# Allocators 

BlackCat Tensor's common-types (Vector, Matrix, Cube) are created using two template arguments. The first argument, is the scalar-type while the second is the allocator. This was designed to mimic `stl` library despite the fact that BCT's allocator's are NOT interchangeable to the standard libraries allocators. 

#### The primary allocators

	Basic_Allocator
	Cuda
	Cuda_Managed

#### Differences between `std` and `BC` allocators. 
1) BC allocators do not accept a template argument for the scalar_type. 
2) BC allocators are composed of entirely static methods. 
3) BC allocators define certain methods that are essential 'boiler plate' for Cuda. IE each allocator (including Basic_Allocator) define `HostToDevice(device_ptr, host_ptr, size)` and `DeviceToHost(host_ptr, device_ptr, size)`. These methods essential to copying to and from device memory. 

Basic_Allocator:
	The standard allocator and default argument.
	The Basic Allocator is comparable to the std::allocator.
	It has no special properties.

Cuda:
	The generic allocator for CUDA. Allocates memory on the device. Memory allocated with Cuda cannot be accessed from the host.

Cuda_Managed:
	Cuda allocates memory vs CudaMallocManaged. The memory is accessible from both host and device. Cuda_Managed is recommended if you are not familiar with Cuda programming.
