/*
 * BC_Tensor_Matrix.h
 *
 *  Created on: Dec 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_MATRIX_H_
#define BC_TENSOR_MATRIX_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
#include "BC_Tensor_Vector.h"
namespace BC {

template<class T, int row, int col, class lib, //default = CPU,
		class LD // default = typename DEFAULT_LD<Inner_Shape<row>>::type
>
class Matrix : public Tensor_Mathematics_Head<T, Matrix<T, row,col, lib>, lib, Static_Inner_Shape<row, col>, typename DEFAULT_LD<Static_Inner_Shape<row, col>>::type> {

	using parent_class      = 		   Tensor_Mathematics_Head<T, Matrix<T, row, col, lib>, lib, Static_Inner_Shape<row, col>, typename DEFAULT_LD<Static_Inner_Shape<row, col>>::type>;
	using this_type         = 		   Matrix<T, row, col, lib, LD>;
	using functor_type      = typename Tensor_FunctorType<T>::type;
	using grandparent_class = typename parent_class::grandparent_class;
	static constexpr Tensor_Shape RANK = MATRIX;


public:

	using parent_class::parent_class;
	Matrix(int rows, int cols) : parent_class({rows, cols}) {}
	template<class U, class alt_LD> Matrix(const Matrix<U, row, col, lib, alt_LD>&  mat) : parent_class(         ) { (*this) = mat; }
	template<		  class alt_LD> Matrix(const Matrix<T, row, col, lib, alt_LD>&  mat) : parent_class(		 ) { (*this) = mat; }
	template<class U, class alt_LD>	Matrix(      Matrix<U, row, col, lib, alt_LD>&& mat) : parent_class(		 ) { (*this) = mat; }
	template<		  class alt_LD>	Matrix(      Matrix<T, row, col, lib, alt_LD>&& mat) : parent_class(mat.array) {				}

	template<class U, class alt_LD>
	Matrix<T, row, col, lib, LD>& operator =(const Matrix<U, row, col, lib, alt_LD>& v) {

		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}
	template<class alt_LD>
	Matrix<T, row, col, lib, LD>& operator =(const typename std::enable_if<
																		grandparent_class::ASSIGNABLE,
																			Matrix<T, row, col, lib, alt_LD>&>::type v)
	{
		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}

	const Matrix<unary_expression_transpose_mat<T, this_type>, col, row, lib, typename DEFAULT_LD<Static_Inner_Shape<col, row>>::type> t() const {
		return Matrix<unary_expression_transpose_mat<T, this_type>, col, row, lib>(*this, this->array);
	}

	      Vector<T, row, lib, LD> operator [] (int index)  {
		return Vector<T, row, lib, LD>(&this->array[index] * row);
	}
	const Vector<T, row, lib, LD> operator [] (int index) const  {
		return Vector<T, row, lib, LD>(&this->array[index] * row);
	}

};


}


#endif /* BC_TENSOR_MATRIX_H_ */
