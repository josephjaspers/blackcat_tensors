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
 *We must define host to device (to match the cuda version of BCT api). CPU version is simply copying
 */

template<class core_lib>
struct CPU_Utility {

	template<class T, class U>
	static void HostToDevice(T* t, U* u, int sz) {
		for (int i = 0; i < sz; ++i)
			t[i] = u[i];
	}
	template<class T, class U>
	static void DeviceToHost(T* t, U* u, int sz) {
		for (int i = 0; i < sz; ++i)
			t[i] = u[i];
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
