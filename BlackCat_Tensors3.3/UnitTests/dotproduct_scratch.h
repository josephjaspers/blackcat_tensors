/*
 * dotproduct_scratch.h
 *
 *  Created on: Mar 20, 2018
 *      Author: joseph
 */

#ifndef DOTPRODUCT_SCRATCH_H_
#define DOTPRODUCT_SCRATCH_H_

#include "../BlackCat_Tensors.h"

namespace BC{

template<class T, int row = 0, int col = 0>
struct dotproduct_impl {

template<class U, class V>
static auto foo(Matrix<T>& out, const Matrix<U>& mat1, const Matrix<V>& mat2)
{
	if (col  != mat2.cols())
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row, col + 1 >::foo(out, mat1, mat2);
	else if (row != mat1.rows())
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row + 1>::foo(out, mat1, mat2);
	else
		return out[row][col] +=* (mat1.row(row) % mat2[col]) && dotproduct_impl<T, row + 1, 0>::foo(out, mat1, mat2);

}
};
template<class T>
struct dotproduct_impl<T, 2, 2> {

	template<class U, class V>
static auto foo(Matrix<T>& out, const Matrix<U>& mat1, const Matrix<V>& mat2, int row = 0, int col = 0)
{
	return out[row][col];
}

};

template<class T>
auto dotproduct(const Matrix<T>& mat1, const Matrix<T>& mat2) {
	Matrix<T> out(mat1.rows(), mat2.cols());

	dotproduct_impl<T,0,0>::foo(out, mat1, mat2);
	return out;
}

}



#endif /* DOTPRODUCT_SCRATCH_H_ */
