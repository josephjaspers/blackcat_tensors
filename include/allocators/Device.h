
/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifdef __CUDACC__
#ifndef BC_ALLOCATOR_DEVICE_H_
#define BC_ALLOCATOR_DEVICE_H_

namespace BC {

class device_tag;

namespace allocators {

template<class SystemTag, class ValueType>
class Allocator;


/// The 'std::allocator' of GPU-allocators.
/// Memory is allocated via 'cudaMalloc'
template<class T>
struct Allocator<device_tag, T> {

    using system_tag = device_tag;		//BC tag

    using value_type = T;
    using pointer = T*;
    using const_pointer = T*;
    using size_type = int;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;

	template<class altT>
	struct rebind { using other = Allocator<device_tag, altT>; };

	template<class U>
	Allocator(const Allocator<device_tag, U>&) {}

	Allocator() = default;
	Allocator(const Allocator&)=default;
	Allocator(Allocator&&) = default;

	Allocator& operator = (const Allocator&) = default;
	Allocator& operator = (Allocator&&) = default;

    T* allocate(std::size_t sz) const {
    	T* data_ptr;
    	BC_CUDA_ASSERT((cudaMalloc((void**) &data_ptr, sizeof(T) * sz)));
        return data_ptr;
    }

    void deallocate(T* data_ptr, std::size_t size) const {
    	BC_CUDA_ASSERT((cudaFree((void*)data_ptr)));
    }

    template<class U> constexpr bool operator == (const Allocator<device_tag, U>&) { return true; }
    template<class U> constexpr bool operator != (const Allocator<device_tag, U>&) { return false; }
};

}
}




#endif /* CUDA_ALLOCATOR_H_ */
#endif
