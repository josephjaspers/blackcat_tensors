/*
 * Tensor_Lion.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef TYPECLASS_FUNCTIONTYPE_H_
#define TYPECLASS_FUNCTIONTYPE_H_
#include <type_traits>
#include "../BC_MetaTemplateFunctions/Adhoc.h"
#include "../BC_MetaTemplateFunctions/Simple.h"

/*
 * Defines T type as either an array (if non class type)
 * or defiens T as a value (given that its a wrapper to an array_tpe)
 */
namespace BC {

template<class T, class enable_mem_ptr = void> //is class or struct (IE expression)
struct Tensor_FunctorType {
	using type = T;
	static constexpr bool supports_utility_functions = false;

};

template<class T>
struct Tensor_FunctorType<T, typename std::enable_if<MTF::isPrimitive<T>::conditional>::type> {
	using type = T*;
	static constexpr bool supports_utility_functions = true;

};

template<class T>
struct Tensor_FunctorType<T, typename std::enable_if<MTF::isExpressionType<T>::conditional>::type> {
	using type = T;
	static constexpr bool supports_utility_functions = false;
};

template<class T>
struct Tensor_FunctorType<T, typename std::enable_if<MTF::isArrayType<T>::conditional>::type> {
	using type = T;
	static constexpr bool supports_utility_functions = false;
};
}

#endif /* TYPECLASS_FUNCTIONTYPE_H_ */
