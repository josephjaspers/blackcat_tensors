/*
 * Tensor_Lion.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_INHERITLV1_FUNCTIONTYPE_H_
#define BC_TENSOR_INHERITLV1_FUNCTIONTYPE_H_

#include <type_traits>
#include "BC_Expression_Base.h"
/*
 * Defines T type as either an array (if non class type)
 * or defiens T as a value (given that its a wrapper to an array_tpe)
 */

template<class T, class enable_mem_ptr = void>
struct Tensor_FunctorType {
	using type = T*;
	static constexpr bool supports_utility = true;	//Can this be a parent class ?
};

template<class T>
struct Tensor_FunctorType<T, typename std::enable_if<std::is_class<T>::value>::type> {
	using type = T;
	static constexpr bool supports_utility = false;  //Can this be a parent class ?
};
template<class T, class deriv>
struct Tensor_FunctorType<T, expression<deriv>> {
	using type = typename expression<deriv>::type;
	static constexpr bool supports_utility = false;  //Can this be a parent class ?
};


namespace BC_Functor_Deleter {
	template<class U, class lib>
	struct destructor {

		static void destroy(U* t) {
			lib::destroy(t);
		}
		static void destroy(U u) {
		}
	};

	template<class T, class U>
	struct isSame {
		static constexpr bool conditional = false;
	};
	template<class T>
	struct isSame<T, T> {
		static constexpr bool conditional = true;
	};
}


#endif /* BC_TENSOR_INHERITLV1_FUNCTIONTYPE_H_ */
