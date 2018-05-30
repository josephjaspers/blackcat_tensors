/*
 * isPrimaryCore.h
 *
 *  Created on: May 20, 2018
 *      Author: joseph
 */

#ifndef ISPRIMARYCORE_H_
#define ISPRIMARYCORE_H_

#include <type_traits>

namespace BC {

/*
 * Determines if a an internal type is a Tensor_Core,
 * 	this is relevant as Tensor_Core's are the only type that has an internal memory_ptr.
 * 	All other classes are just expressions (this is semi-not true for dotproduct, reshape, and chunk)
 *
 */

class BC_Type;
template<class> class Core;
template<class> class lambda_array;
template<class, int> class stack_array;
template<int> class Shape;
template<int, class, class> class t_shape;

template<class T> struct isPrimaryCore { static constexpr bool conditional = false; };
template<class T> struct isPrimaryCore<Core<T>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool pCore_b = isPrimaryCore<T>::conditional;

//determines if the type is a valid-indexable tensor_shape
template<class T> 	struct BlackCat_Shape 						{ static constexpr bool conditional = false; };
template<> 			struct BlackCat_Shape<int*> 				{ static constexpr bool conditional = true; };

template<>			struct BlackCat_Shape<const int*> 			{ static constexpr bool conditional = true; };
template<>			struct BlackCat_Shape<const int* const&> 	{ static constexpr bool conditional = true; };

template<class T> 	struct BlackCat_Shape<lambda_array<T>> 		{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<stack_array<int,x>> 	{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<Shape<x>> 			{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<Shape<x>&> 			{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<const Shape<x>> 			{ static constexpr bool conditional = true; };
template<int x> 	struct BlackCat_Shape<const Shape<x>&> 			{ static constexpr bool conditional = true; };
template<int x, class i, class o> 	struct BlackCat_Shape<t_shape<x, i, o>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool is_shape = BlackCat_Shape<T>::conditional;

}



#endif /* ISPRIMARYCORE_H_ */
