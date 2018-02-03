/*
 * Matrix.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_MATRIX_H
#define BC_MATRIX_H

#include "Vector.h"
#include "Tensor_Base.h"


namespace BC {
template<class T, class Mathlib>
class Matrix : public Tensor_Base<T, Matrix<T, Mathlib>, Mathlib>
{
	using parent_class = Tensor_Base<T, Matrix<T, Mathlib>, Mathlib>;
	using _int = typename parent_class::subAccess_int;
	using __int = typename parent_class::force_evaluation_int;
	template<class,class>
	friend class Vector;

public:
	static constexpr int RANK() { return 2; }

	using parent_class::operator=;
	using parent_class::parent_class;

	Matrix(int rows, int cols = 1) : parent_class({rows, cols}) {}
	Matrix(Matrix<T, Mathlib>&& mat) : parent_class(mat.expression_packet(), mat.data()) {}

	template<class U> Matrix(const Matrix<U, Mathlib>& mat) : parent_class(mat.expression_packet()) { Mathlib::copy(this->data(), mat.data(), this->size()); }
	Matrix(const Matrix<T, Mathlib>& mat) : parent_class(mat.expression_packet()) { Mathlib::copy(this->data(), mat.data(), this->size()); }


	Vector<T, Mathlib> operator[] (_int i) {
		return (Vector<T, Mathlib>(this->accessor_packet(), &this->array[i]));
	}
	const Vector<T, Mathlib> operator[] (_int i) const {
		return Vector<T, Mathlib>(this->accessor_packet(), &this->array[i]);
	}

	auto operator [] (__int i) const {
		return this->data()[i];
	}

	const Matrix<unary_expression_transpose<typename MTF::determine_scalar<T>::type, Matrix<T, Mathlib>>, Mathlib> t() const {
		return Matrix<unary_expression_transpose<typename MTF::determine_scalar<T>::type, Matrix<T, Mathlib>>, Mathlib>(this->transpose_packet(), *this);
	}

	Matrix<T, Mathlib>& operator = (const Matrix<T, Mathlib>& mat) {
		this->assert_same_size(mat);
		Mathlib::copy(this->data(), mat.data(), this->size());
		return this->asBase();
	}
	template<class U>
	Matrix<T, Mathlib>& operator = (const Matrix<U, Mathlib>& mat) {
		this->assert_same_size(mat);
		Mathlib::copy(this->data(), mat.data(), this->size());
		return this->asBase();
	}

};

} //End Namespace BC

#endif /* MATRIX_H */
