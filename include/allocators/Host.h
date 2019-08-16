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

template<class SystemTag, class ValueType>
class Allocator;

/// Comparable to 'std::allocator.'
template<class T>
struct Allocator<host_tag, T> {

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
	struct rebind { using other = Allocator<host_tag, altT>; };

	template<class U>
	Allocator(const Allocator<host_tag, U>&) {}
	Allocator() = default;
	Allocator(const Allocator&) = default;
	Allocator(Allocator&&) = default;

	Allocator& operator = (const Allocator&) = default;
	Allocator& operator = (Allocator&&) = default;

    T* allocate(int size) {
        return new T[size];
    }

    void deallocate(T* t, BC::size_t  size) {
        delete[] t;
    }

    template<class U> constexpr bool operator == (const Allocator<host_tag, U>&) const { return true; }
    template<class U> constexpr bool operator != (const Allocator<host_tag, U>&) const { return false; }
};

}
}




#endif /* BASIC_ALLOCATOR_H_ */
