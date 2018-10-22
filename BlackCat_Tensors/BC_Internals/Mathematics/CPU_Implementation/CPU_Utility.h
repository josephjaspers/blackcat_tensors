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
struct CPU_Utility {

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
