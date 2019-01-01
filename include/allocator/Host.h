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
struct Host {

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

    void deallocate(T* t, BC::size_t  size) {
        delete[] t;
    }

    constexpr bool operator == (const Host&) { return true; }
    constexpr bool operator != (const Host&) { return false; }
};

template<class T, int id=0, class derived=void>
struct Host_Unsynced_Memory_Stack {
	static constexpr int ID = id; //The ID of the pool.

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

	static std::size_t pool_size;
	static std::size_t pool_index;
	static T* memory_pool;

	static void initialize_pool(std::size_t init_size) {
		if (memory_pool) {
			throw ("Memory pool already created");
		}
		memory_pool = new T[init_size];
	}

	static void destruct_pool() {
		delete[] memory_pool;
	}

    T* allocate(std::size_t size) {
    	if (pool_index + size >= pool_size) {
    		throw ("Memory pool exhausted");
    	}
    	T* memptr = & memory_pool[pool_index];
    	pool_index += size;

    	return memptr;
    }

    void deallocate(T* t, BC::size_t  size) {
    	pool_index -= size;
    }

    constexpr bool operator == (const Host_Unsynced_Memory_Stack&) { return true;  }
    constexpr bool operator != (const Host_Unsynced_Memory_Stack&) { return false; }
};
}
}




#endif /* BASIC_ALLOCATOR_H_ */
