/*
 * CPU_Constants.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef CPU_CONSTANTS_H_
#define CPU_CONSTANTS_H_

namespace BC {

template<class core_lib>
struct CPU_Constants {

	template<class U, class T, class V>
	static void scalar_mul(U eval, T* a, V* b) {
		*eval = a[0] * b[0];
	}
	template<class U, class T, class V>
	static void scalar_mul(U eval, T a, V* b) {
		*eval = a * b[0];
	}
	template<class U, class T, class V>
	static void scalar_mul(U eval, T* a, V b) {
		*eval = a[0] * b;
	}
	template<class U, class T, class V>
	static void scalar_mul(U eval, T a, V b) {
		*eval = a * b;
	}

	template<class T>
	static T* static_initialize(T value) {
		return new T(value);
	}
	static constexpr float s1 = 1;
	static constexpr float sn1 = -1;
	static constexpr float s0 = 0;

	static constexpr double d1 = 1;
	static constexpr double dn1 = -1;
	static constexpr double d0 = 0;




};


}


#endif /* CPU_CONSTANTS_H_ */
