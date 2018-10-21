/*
 * Utilities_CPU.h
 *
 *  Created on: May 6, 2018
 *      Author: joseph
 */

#ifndef UTILITIES_CPU_H_
#define UTILITIES_CPU_H_

namespace BC {

/*
 * Defines methods relating to internal creation/destruction and I/O
 */

template<class core_lib>
struct CPU_Allocator {

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
	template<class T, class U>
	static void HostToDevice(T* device_ptr, U* host_ptr, int size=1) {
		core_lib::copy(device_ptr, host_ptr, size);
	}
	template<class T, class U>
	static void DeviceToHost(T* host_ptr, U* device_ptr, int size=1) {
		core_lib::copy(host_ptr, device_ptr, size);
	}
	template<typename T>
	static void deallocate(T* t) {
		delete[] t;
	}
	template<typename T>
	static void deallocate(T t) {
		//empty
	}
	template<class T, class is, class os>
	static void print(const T array_ptr, const is inner_shape, const os outer_shape, int numb_dimensions, int print_gap_length) {
		BC::IO::print(array_ptr, inner_shape, outer_shape, numb_dimensions, print_gap_length);
	}

	template<class T, class is, class os>
	static void printSparse(const T array_ptr, const is inner_shape, const os outer_shape, int numb_dimensions, int print_gap_length) {
		BC::IO::printSparse(array_ptr, inner_shape, outer_shape, numb_dimensions, print_gap_length);
	}
};
}



#endif /* UTILITIES_CPU_H_ */
