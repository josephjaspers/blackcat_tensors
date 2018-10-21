/*
 * CPU_Constants.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef CPU_CONSTANTS_H_
#define CPU_CONSTANTS_H_

namespace BC {
namespace cpu_impl {

template<class T, class enabler = void>
struct get_value {
	static auto impl(T scalar) {
		return scalar;
	}
};
template<class T>
struct get_value<T, std::enable_if_t<!std::is_same<decltype(std::declval<T&>()[0]), void>::value>>  {
	static auto impl(T scalar) {
		return scalar[0];
	}
};
} //end of cpu_impl namespace


template<class core_lib>
struct CPU_Constants {

	template<class U, class T, class V>
	static void scalar_mul(U& eval, T a, V b) {
		eval = cpu_impl::get_value<T>::impl(a) * cpu_impl::get_value<V>::impl(b);
	}

	template<class T>
	static T static_allocate(T value) {
		return T(value);
	}
};


}


#endif /* CPU_CONSTANTS_H_ */
