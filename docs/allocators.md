# Allocators 

BlackCat Tensor's common-types (Vector, Matrix, Cube) are created using two template arguments. The first argument, is the scalar-type while the second is the allocator. This was designed to mimic `stl` library.

	
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
	
	
#### Additional Tags of BlackCat Allocators

	BlackCat Allocators utilize 2 additional tags to achieve various modifiability. 
	

	`system_tag`

	The system_tag, may be either BC::host_tag or BC::device_tag.
	`system_tag` informs if the memory is allocated on the GPU or CPU. 
	This enables compile time checking to make sure expressions are computed in the apropriate manner as well
	as enabling querying for apropriate default behavior across the CPU and GPU. 

	If system_tag is not supplied, it is defaulted to 'host_tag'.


	`propagate_on_temporary_construction` 

	The 'propagate_on_temporary_construction' enables the user to specify the behavior of memory allocation
	when temporaries are needed. 
	
	if 'propagate_on_temporary_construction' is set to 'std::false_type', the default allocator is used.
	if 'propagate_on_temporary_construction' is set to 'std::true_type', the same allocator is used,
		the allocator will use "select_on_container_copy_construction" to generate the new allocator. 
	
	if 'propagate_on_temporary_construction' is not defined it is defaulted to be 'std::false_type' 


#### Choosing an allocator (example):

```cpp
BC::Matrix<float> mat; 			    	   //defaults to Basic_Allocator<float>
BC::Matrix<float, BC::Basic_Allocator<float>> mat; //identical to above   
BC::Matrix<float, BC::Cuda<float>> mat;	    	   //allocates memory on the GPU 
BC::Matrix<float, BC::Cuda_Managed<float>> mat;    //allocates memory on the GPU but data transfer is managed automatically. 
```
