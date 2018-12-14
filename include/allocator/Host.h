/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BASIC_ALLOCATOR_H_
#define BASIC_ALLOCATOR_H_

namespace BC {

class host_tag;

namespace allocator {

template<class T, class derived=void>
struct Host : AllocatorBase<std::conditional_t<std::is_void<derived>::value, Host<T>, derived>> {

    using system_tag = host_tag;	//BC tag
    using propagate_to_expressions = std::false_type; //BC tag

    using value_type = T;
    using pointer = T*;
    using const_pointer = T*;
    using size_type = int;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;
    using is_always_equal = std::true_type;


    T* allocate(int size) {
        return new T[size];
    }

    void deallocate(T* t) {
        delete[] t;
    }

    constexpr bool operator == (const Host&) { return true; }
    constexpr bool operator != (const Host&) { return false; }


};
}
}




#endif /* BASIC_ALLOCATOR_H_ */
