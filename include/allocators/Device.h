
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

template<class T>
struct Device {

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
	struct rebind { using other = Device<altT>; };

	template<class U>
	Device(const Device<U>&) {}

	Device() = default;
	Device(const Device&)=default;
	Device(Device&&) = default;

	Device& operator = (const Device&) = default;
	Device& operator = (Device&&) = default;

    T* allocate(std::size_t sz) const {
    	T* data_ptr;
    	BC_CUDA_ASSERT((cudaMalloc((void**) &data_ptr, sizeof(T) * sz)));
        return data_ptr;
    }

    void deallocate(T* data_ptr, std::size_t size) const {
    	BC_CUDA_ASSERT((cudaFree((void*)data_ptr)));
    }

    constexpr bool operator == (const Device&) { return true; }
    constexpr bool operator != (const Device&) { return false; }


};

}
}




#endif /* CUDA_ALLOCATOR_H_ */
#endif
