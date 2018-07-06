/*
 * isPrimaryArray.h
 *
 *  Created on: May 20, 2018
 *      Author: joseph
 */

#ifndef ISPRIMARYCORE_H_
#define ISPRIMARYCORE_H_

#include <type_traits>

namespace BC {

/*
 * Determines if a an internal type is a Tensor_Array,
 * 	this is relevant as Tensor_Array's are the only type that has an internal memory_ptr.
 * 	All other classes are just expressions (this is semi-not true for gemm, reshape, and chunk)
 *
 */

namespace internal {
template<int, class, class> class Array;
}
template<int, class, class> class lambda_array;
template<int, class> class stack_array;
template<int> class Shape;

template<class T> struct isPrimaryArray { static constexpr bool conditional = false; };
template<int d, class T, class ml> struct isPrimaryArray<internal::Array<d,T,ml>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool is_array_core() { return isPrimaryArray<T>::conditional; }

//determines if the type is a valid-indexable tensor_shape
template<class T> 	struct BlackCat_Shape 						{ static constexpr bool conditional = false; };
template<int x, class S, class T> 	struct BlackCat_Shape<lambda_array<x, S, T>> 		{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<stack_array<x, int>> 	{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<Shape<x>> 			{ static constexpr bool conditional = true; };
template<class T> static constexpr bool is_shape = BlackCat_Shape<std::decay_t<T>>::conditional;

}



#endif /* ISPRIMARYCORE_H_ */
