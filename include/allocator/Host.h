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

struct Host {

    using system_tag = host_tag;

    template<typename T>
    static T*& allocate(T*& internal_mem_ptr, int size) {
        internal_mem_ptr = new T[size];
        return internal_mem_ptr;
    }

    template<typename T>
    static void deallocate(T* t) {
        delete[] t;
    }


};
}
}




#endif /* BASIC_ALLOCATOR_H_ */
