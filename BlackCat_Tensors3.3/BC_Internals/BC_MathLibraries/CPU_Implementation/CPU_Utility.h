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
 * Defines methods relating to data creation/destruction and I/O
 */

template<class core_lib>
struct CPU_Utility {

	template<typename T>
	static T*& initialize(T*& t, int sz) {
		t = new T[sz];
		return t;
	}
	template<typename T>
	static T*& unified_initialize(T*& t, int sz) {
		t = new T[sz];
		return t;
	}
	template<class T, class U>
	static void HostToDevice(T* t, U* u, int size) {
		core_lib::copy(t, u, size);
	}
	template<class T, class U>
	static void DeviceToHost(T* t, U* u, int size) {
		core_lib::copy(t, u, size);
	}
	template<typename T>
	static void destroy(T* t) {
		delete[] t;
	}
	template<class T, class is, class os>
	static void print(const T ary, const is inner, const os outer, int order, int print_length) {
		BC::print(ary, inner, outer, order, print_length);
	}

	template<class T, class is, class os>
	static void printSparse(const T ary, const is inner, os outer, int order, int print_length) {
		BC::printSparse(ary, inner, outer, order, print_length);
	}
};
}



#endif /* UTILITIES_CPU_H_ */
