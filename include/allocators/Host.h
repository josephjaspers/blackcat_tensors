/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_ALLOCATOR_HOST_H_
#define BC_ALLOCATOR_HOST_H_

namespace BC {

class host_tag;

namespace allocators {


template<class T>
struct Host {

    using system_tag = host_tag;	//BC tag

    using value_type = T;
    using pointer = T*;
    using const_pointer = T*;
    using size_type = int;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;

	template<class altT>
	struct rebind { using other = Host<altT>; };

	template<class U>
	Host(const Host<U>&) {}
	Host() = default;
	Host(const Host&) = default;
	Host(Host&&) = default;

	Host& operator = (const Host&) = default;
	Host& operator = (Host&&) = default;

    T* allocate(int size) {
        return new T[size];
    }

    void deallocate(T* t, BC::size_t  size) {
        delete[] t;
    }

    constexpr bool operator == (const Host&) { return true; }
    constexpr bool operator != (const Host&) { return false; }
};

}
}




#endif /* BASIC_ALLOCATOR_H_ */
