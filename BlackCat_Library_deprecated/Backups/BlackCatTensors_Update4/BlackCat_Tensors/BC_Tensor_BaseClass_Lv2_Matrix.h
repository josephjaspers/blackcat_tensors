/*
 * BC_Tensor_BaseClass_Lv2_Matrix.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_BASECLASS_LV2_MATRIX_H_
#define BC_TENSOR_BASECLASS_LV2_MATRIX_H_

#include "BC_Tensor_BaseClass_Lv1_Vector.h"
#include "BC_Expression_Unary_Transpose.h"

template<class T, class ml, int ... dimensions>
class Matrix : Tensor_Queen<T, ml, dimensions...> {

};

template<class T, class ml, int rows, int cols>
class Matrix<T, ml, rows, cols> : public Tensor_Queen<T, ml, rows, cols> {
public:
	Matrix<T, ml, rows, cols> () = default;
	Matrix<T, ml, rows, cols> (T* ary) : Tensor_Queen<T, ml, rows, cols>(ary) {}


	template<class... params>
	Matrix<T, ml, rows, cols> (const params&... p) : Tensor_Queen<T, ml, rows, cols>(p...) {}


	Vector<T, ml, rows> operator [] (int index) {
		return Vector<T, ml, rows>(&this->data()[index * this->rows()]);
	}

	const Vector<T, ml, rows> operator [] (int index) const {
		return Vector<T, ml, rows>(&this->data()[index * this->rows()]);
	}

	const Matrix<transpose_expression<T, ml, cols, rows>, ml, cols, rows> t() const {
		return Matrix<transpose_expression<T, ml, cols, rows>, ml, cols, rows>(const_cast<T*>(this->array));
	}


	Matrix<T, ml, rows, cols>& operator =(const Tensor_Ace<T, ml, rows, cols>& tens) {
		ml::copy(this->data(), tens.data(), this->size());
		return *this;
	}

	template<class U>
	Matrix<T, ml, rows, cols>& operator =(const Tensor_Ace<U, ml, rows, cols>& tens) {
		ml::copy(this->data(), tens.data(), this->size());
		return *this;
	}

	Matrix<T, ml, rows, cols>& operator =(const Matrix<T, ml, rows, cols>& tens) {
		ml::copy(this->data(), tens.data(), this->size());
		return *this;
	}

	template<class U>
	Matrix<T, ml, rows, cols>& operator =(const Matrix<U, ml, rows, cols>& tens) {
		ml::copy(this->data(), tens.data(), this->size());
		return *this;
	}
};

#endif /* BC_TENSOR_BASECLASS_LV2_MATRIX_H_ */
