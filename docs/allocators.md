# Allocators 

BlackCat Tensor's common-types (Vector, Matrix, Cube) are created using two template arguments. The first argument, is the scalar-type while the second is the allocator. This was designed to mimic the `stl` library.

	
#### Overview

*Basic_Allocator*:

	The default allocator. 
	The Basic_Allocator is comparable to the std::allocator.
	It has no special properties.

*Cuda_Allocator*:

	The generic allocator for CUDA. 
	Allocates memory on the device. 
	Memory allocated with Cuda cannot be accessed from the host.

*Cuda_Managed*:

	Cuda_Managed allocates memory via CudaMallocManaged. 
	The memory is accessible from both host and device. 
	Cuda_Managed is recommended if you are not familiar with Cuda programming.
	
	
#### Additional Tags of BlackCat Allocators

	BlackCat Allocators utilizes the 'system_tag' to differentiate between host and device allocators.
	
	`system_tag`

	The system_tag, may be either BC::host_tag or BC::device_tag.
	
```cppp
using system_tag = BC::device_tag; //Will inform BC_Tensors your allocator allocates memory on the GPU.
```	
	`system_tag` informs if the memory is allocated on the GPU or CPU. 
	Tensors constructed with 'device' allocators will have their computations run on the GPU.

	If system_tag is not supplied, it is defaulted to 'host_tag'.


#### Choosing an allocator (example):

```cpp
BC::Matrix<float> mat; 			    	   //defaults to Basic_Allocator<float>
BC::Matrix<float, BC::Basic_Allocator<float>> mat; //identical to above   
BC::Matrix<float, BC::Cuda_Allocator<float>> mat;  //allocates memory on the GPU 
BC::Matrix<float, BC::Cuda_Managed<float>> mat;    //allocates memory on the GPU but data transfer is managed automatically. 
```

#### Fancy Allocators:

	BlackCat_Tensors supports some more advanced allocators found in the namespace BC::allocators::fancy.
	Allocators in the 'fancy' namespace have more advanced properties than the aforementioned allocators. 
	They're generally used internally for performance-critical components of the BCT.
	
```cpp
	template<class ValueType, class SystemTag>
	struct Polymorphic_Allocator;
	
	//The Polymorphic allocator function similarly to the C++17 std::pmr::polymorphic_allocator. 
	//It accepts another allocator during its construction and uses Virtual-Calls to enable changing the underlying allocator at run time. 

	template<class ValueType, class SystemTag>
	struct Workspace;

	//The Workspace allocator works as un unsynchronized-memory stack. 
	//It;s deallocations must be in reverse order of it's allocations.
```
