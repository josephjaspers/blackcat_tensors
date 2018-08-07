///*
// * Matrix.h
// *
// *  Created on: Dec 30, 2017
// *      Author: joseph
// */
//
//#ifndef BC_MATRIX_H
//#define BC_MATRIX_H
//
//#include "BC_Tensor_Base/Tensor_Base.h"
//
//namespace BC {
//
//template<class T, class Mathlib>
//class Matrix : public Tensor_Base<Matrix<T, Mathlib>> {
//
//	using parent_class = Tensor_Base<Matrix<T, Mathlib>>;
//
//public:
//
//	__BCinline__ static constexpr int DIMS() { return 2; }
//	using parent_class::operator=;
//	using parent_class::operator[];
//	using parent_class::operator();
//	//constructors---------------------------------------------------
//
//	Matrix(const Matrix&  v) : parent_class(v) {}
//	Matrix(		 Matrix&& v) : parent_class(v) {}
//
//	explicit Matrix(int rows = 0, int cols = 1) : parent_class(Shape<2>(rows, cols)) {}
//	explicit Matrix(Shape<DIMS()> shape) : parent_class(shape)  {}
//
//	template<class... params>
//	Matrix(const params&... p) : parent_class(p...) {}
//
//	//copy_operators--------------------------------------------------
//
//	Matrix& operator =(const Matrix& t)  { return parent_class::operator=(t); }
//	Matrix& operator =(	     Matrix&& t) { return parent_class::operator=(std::move(t)); }
//
//	template<class U>
//	Matrix& operator = (const Matrix<U, Mathlib>& t) { return parent_class::operator=(t); }
//};
//} //End Namespace BC
//
//#endif /* MATRIX_H */
