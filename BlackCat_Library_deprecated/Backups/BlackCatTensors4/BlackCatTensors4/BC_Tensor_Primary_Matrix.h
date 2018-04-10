/*
 * BC_Tensor_Primary_Matrix.h
 *
 *  Created on: Nov 25, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_PRIMARY_MATRIX_H_
#define BC_TENSOR_PRIMARY_MATRIX_H_

#include "BC_Expression_Transposer.h"
#include "BC_Tensor_AssistClass_DotProduct.h"
#include "BC_Tensor_Primary_Vector.h"
#include "BC_Tensor_Super_Jack.h"

template<class T, class ml, int ... dims>
class Matrix;

template<class T, class ml, int rows, int cols>
class Matrix<T, ml, rows, cols> : public Tensor_Jack<T, ml, rows, cols> {
public:

	using functor_type = typename Tensor_Jack<T, ml, rows, cols>::functor_type;

	template<class, class, int...>
	friend class Matrix;

	//constructors
	Matrix<T, ml, rows, cols>() {
		ml::initialize(this->array, this->size());
	}
	Matrix<T, ml, rows, cols>(T* ary) {
		this->array = ary;
	}

	//returns a row vector

	transpose_expression<functor_type, ml, cols, rows> t() const {
		return transpose_expression<functor_type, ml, cols, rows>(this->data());
	}

	template<typename U>
	Matrix<T, ml, rows, cols> operator =(const Tensor_Queen<U, ml, rows, cols>& tk) {
		ml::copy(this->data(), tk.data(), this->size());
		return *this;
	}

	Vector<T, ml, rows> operator [](int index) {
		return Vector<T, ml, rows>(&this->array[this->rows() * index]);
	}

	const Vector<T, ml, rows> operator [](int index) const {
		return Vector<T, ml, rows>(&this->array[this->rows() * index]);
	}
};


//Specializations
//template<class T, class ml, int row>
//class Matrix<T, ml, row, 1> : public Vector<T, ml, row> {
//public:
//
//	template<class type>
//	Matrix<T, ml, row, 1>& operator = (const type& t) {
//		Vector<T, ml, row>::operator=(t);
//		return * this;
//	}
//
//};
//template<class T, class ml, int col>
//class Matrix<T, ml, 1, col> : public Vector<T, ml, 1, col> {
//public:
//
//	template<class type>
//	Matrix<T, ml, col, 1>& operator = (const type& t) {
//		Vector<T, ml, 1, col>::operator=(t);
//		return * this;
//	}
//
//};

#endif /* BC_TENSOR_PRIMARY_MATRIX_H_ */
