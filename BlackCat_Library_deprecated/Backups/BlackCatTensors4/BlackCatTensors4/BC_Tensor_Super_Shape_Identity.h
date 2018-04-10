/*
 * BC_Tensor_Super_Shape_Identity.h
 *
 *  Created on: Nov 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_SHAPE_IDENTITY_H_
#define BC_TENSOR_SUPER_SHAPE_IDENTITY_H_

#include "BC_InternalIncludes.h"

template<class T, class ml, int... dimensions>
struct BC_Identity {
	using type = Tensor<T, ml, dimensions...>;
};

template<class T, class ml, int rows>
struct BC_Identity<T, ml, rows> {
	using type = Vector<T, ml, rows>;
};

template<class T, class ml, int rows, int cols>
struct BC_Identity<T, ml, rows, cols> {
	using type = Matrix<T, ml, rows, cols>;
};


#endif /* BC_TENSOR_SUPER_SHAPE_IDENTITY_H_ */
