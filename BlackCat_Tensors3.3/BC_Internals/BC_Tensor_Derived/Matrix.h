/*
 * Matrix.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_MATRIX_H
#define BC_MATRIX_H
#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {

template<class T, class Mathlib>
class Matrix : public Tensor_Base<Matrix<T, Mathlib>> {

	using parent_class = Tensor_Base<Matrix<T, Mathlib>>;

public:

	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return 2; }

	explicit Matrix(int rows = 0, int cols = 1) : parent_class(Shape<2>(rows, cols)) {}
	Matrix(const Matrix&  v) : parent_class(v) {}
	Matrix(		 Matrix&& v) : parent_class(v) {}
	Matrix(const Matrix&& v) : parent_class(v) {}

	template<class U> 		  Matrix(const Matrix<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Matrix(	   Matrix<U, Mathlib>&& t) : parent_class(t) {}

	Matrix& operator =(const Matrix& t)  { return parent_class::operator=(t); }
	Matrix& operator =(const Matrix&& t) { return parent_class::operator=(std::move(t)); }
	Matrix& operator =(	     Matrix&& t) { return parent_class::operator=(std::move(t)); }
	template<class U>
	Matrix& operator = (const Matrix<U, Mathlib>& t) { return parent_class::operator=(t); }

	const Matrix<unary_expression<typename parent_class::functor_type, transpose>, Mathlib> t() const {
		return Matrix<unary_expression<typename parent_class::functor_type, transpose>, Mathlib>(this->data());
	}

private:

	template<class> friend class Tensor_Base;
	template<class> friend class Tensor_Operations;
	template<class... params> Matrix(const params&... p) : parent_class(p...) {}

};

} //End Namespace BC

#endif /* MATRIX_H */
