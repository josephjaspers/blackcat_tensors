/*
 * Basic_Allocator.h
 *
 *  Created on: Oct 22, 2018
 *      Author: joseph
 */

#ifndef BASIC_ALLOCATOR_H_
#define BASIC_ALLOCATOR_H_

namespace BC {

class CPU;

namespace module {
namespace stl  {

struct Basic_Allocator : CPU {

	template<typename T>
	static T*& allocate(T*& internal_mem_ptr, int size) {
		internal_mem_ptr = new T[size];
		return internal_mem_ptr;
	}
	template<typename T>
	static T*& unified_allocate(T*& intenral_mem_ptr, int size) {
		intenral_mem_ptr = new T[size];
		return intenral_mem_ptr;
	}
	template<typename T>
	static void deallocate(T* t) {
		delete[] t;
	}
	template<typename T>
	static void deallocate(T t) {
		//empty
	}

};
}
}
}



#endif /* BASIC_ALLOCATOR_H_ */
