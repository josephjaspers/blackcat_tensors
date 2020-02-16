
/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_ALLOCATOR_DEVICE_H_
#define BC_ALLOCATOR_DEVICE_H_

namespace bc {

class device_tag;
class host_tag;

namespace allocators {

template<class ValueType, class SystemTag>
struct Basic_Allocator_Base
{
	using system_tag = SystemTag;	//BC tag
	using value_type = ValueType;
	using pointer = value_type*;
	using const_pointer = value_type*;
	using size_type = int;
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_move_assignment = std::false_type;
	using propagate_on_container_swap = std::false_type;
	using is_always_equal = std::true_type;

	template<class U>
	bool operator == (const Basic_Allocator_Base<U, SystemTag>&) const {
		return true;
	}

	template<class U>
	bool operator != (const Basic_Allocator_Base<U, SystemTag>&) const {
		return false;
	}
};

template<class ValueType, class SystemTag>
class Allocator;

/// Comparable to 'std::allocator.'
template<class T>
struct Allocator<T, host_tag>: Basic_Allocator_Base<T, host_tag> {

	template<class altT>
	struct rebind { using other = Allocator<altT, host_tag>; };

	template<class U>
	Allocator(const Allocator<U, host_tag>&) {}
	Allocator() = default;

	T* allocate(int size) {
		return new T[size];
	}

	void deallocate(T* t, bc::size_t  size) {
		delete[] t;
	}
};


#ifdef __CUDACC__

/// The 'std::allocator' of GPU-allocators. Memory is allocated via 'cudaMalloc'
template<class T>
struct Allocator<T, device_tag>: Basic_Allocator_Base<T, device_tag> {

	template<class altT>
	struct rebind { using other = Allocator<altT, device_tag>; };


	template<class U>
	Allocator(const Allocator<U, device_tag>&) {}
	Allocator() = default;

	T* allocate(std::size_t sz) const
	{
		T* data_ptr;
		unsigned memsz = sizeof(T) * sz;

		BC_CUDA_ASSERT((cudaMalloc((void**) &data_ptr, memsz)));
		return data_ptr;
	}

	void deallocate(T* data_ptr, std::size_t size) const {
		BC_CUDA_ASSERT((cudaFree((void*)data_ptr)));
	}
};

template<class T>
struct Device_Managed:
	Allocator<T, device_tag>
{
	using Allocator<T, device_tag>::Allocator;
	static constexpr bool managed_memory = true;

	template<class altT>
	struct rebind { using other = Device_Managed<altT>; };

	T* allocate(bc::size_t sz)
	{
		T* data = nullptr;
		unsigned memsz = sizeof(T) * sz;

		BC_CUDA_ASSERT((cudaMallocManaged((void**) &data, memsz)));
		BC_CUDA_ASSERT((cudaDeviceSynchronize()));
		return data;
	}
};


#endif /* #ifdef __CUDACC__ */


}
}

#endif
