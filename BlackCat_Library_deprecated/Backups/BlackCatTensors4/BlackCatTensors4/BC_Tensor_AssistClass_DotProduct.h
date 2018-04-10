/*
 * BC_Tensor_Super_Assist_Class_DotProduct.h
 *
 *  Created on: Nov 29, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_ASSISTCLASS_DOTPRODUCT_H_
#define BC_TENSOR_ASSISTCLASS_DOTPRODUCT_H_

#include "BC_Tensor_Super_Queen.h"

template<class T, class ml, int... dimensions>
class BC_DotProduct {

};

/*
 * CRTP  Matrix and Vector classes
 */
template<class T, class ml, int... dimensions>
class Vector;
template<class T, class ml, int... dimensions>
class Matrix;

template<class T, class ml, int rows, int cols>
class BC_DotProduct<T, ml, rows, cols> : public virtual Tensor_Queen<T, ml, rows, cols>{

public:

	Vector<T, ml, rows> operator * (const Vector<T, ml, cols>& a) {
		Vector<T, ml, rows> mulVec;
		mulVec.zero();
		ml::matmulBLAS(mulVec.array, rows,  cols, 1);
		return mulVec;
	}

	template<int rv_cols>
	Matrix<T, ml, rows> operator * (const Matrix<T, ml, cols, rv_cols>& a) {
		Matrix<T, ml, rows> mulMat;
		mulMat.zero();
		ml::matmulBLAS(mulMat.array, rows, cols, a.array, rv_cols);
		return mulMat;
	}


};



#endif /* BC_TENSOR_ASSISTCLASS_DOTPRODUCT_H_ */
