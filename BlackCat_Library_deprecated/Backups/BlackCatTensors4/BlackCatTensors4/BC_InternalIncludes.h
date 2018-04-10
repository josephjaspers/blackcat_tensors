/*
 * BC_InternalIncludes.h
 *
 *  Created on: Nov 29, 2017
 *      Author: joseph
 */

#ifndef BC_INTERNALINCLUDES_H_
#define BC_INTERNALINCLUDES_H_

template<class T, class ml, bool>
class Scalar;

template<class T, class ml, int ... dimension>
class Vector;

template<class T, class ml, int ... dimension>
class Matrix;

template<class T, class ml, int ... dimension>
class Cube;

template<class T, class ml, int ... dimension>
class Tensor;

//namespace BC_Shape_Identity {
//
//	template<int...>
//	struct validate_dimensions;
//
//	template<int...>
//	struct _dim;
//
//	template<template<int ... params> class list>
//	struct reverse {
//
//
//		template<class reverse_list, int ... list>
//		struct reverse_helper;
//
//		template<template<int...> class reversed, int... r_list, class front, int... list>
//		struct reverse_helper<reversed<r_list...>, front, list...> {
//			using type = reverse_helper<reversed<front, r_list...>, list...>::type;
//		};
//
//		template<template<int...> class reversed, int... r_list>
//		struct reverse_helper<reversed<>, r_list...> {
//			using type = reversed<r_list>;
//		};
//
//	};
//
//	template<class T, class ml, int ... dimensions>
//	struct _shape {
//		using type = Tensor<T, ml, dimensions...>;
//	};
//
//	template<class T, class ml, int rows, int cols>
//	struct _shape<T, ml, rows, cols> {
//		using type = Matrix<T, ml, rows, cols>;
//	};
//
//	template<class T, class ml, int rows>
//	struct _shape<T, ml, rows> {
//		using type = Vector<T, ml, rows>;
//	};
//
//}

#endif /* BC_INTERNALINCLUDES_H_ */
