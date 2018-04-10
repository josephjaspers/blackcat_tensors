/*
 * BC_Tensor_Cube.h
 *
 *  Created on: Dec 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_CUBE_H_
#define BC_TENSOR_CUBE_H_




#include "../BlackCat_Internal_GlobalUnifier.h"
#include "BC_Tensor_Vector.h"
namespace BC {
template<class T, int row, int col, int depth, class lib, //default = CPU,
		class LD // default = typename DEFAULT_LD<Inner_Shape<row>>::type
>
class Cube : public Tensor_Mathematics_Head<T, Cube<T, row,col, depth, lib>, lib, Static_Inner_Shape<row, col, depth>, typename DEFAULT_LD<Static_Inner_Shape<row, col, depth>>::type> {

	using functor_type = typename Tensor_FunctorType<T>::type;
	using parent_class = Tensor_Mathematics_Head<T, Cube<T, row, col, depth, lib>, lib, Static_Inner_Shape<row, col, depth>, typename DEFAULT_LD<Static_Inner_Shape<row, col, depth>>::type>;
	using grandparent_class = typename parent_class::grandparent_class;
	using this_type = Cube<T, row, col, depth, lib>;
	static constexpr Tensor_Shape RANK = CUBE;

public:

	using parent_class::parent_class;
	Cube(int rows, int cols, int depths) : parent_class({rows, cols, depths}) {}
	template<class U, class alt_LD>	Cube(const Cube<U, row, col, depth, lib>&  vec) : parent_class() { (*this) = vec; }
	template<class U, class alt_LD>	Cube(      Cube<U, row, col, depth, lib>&& vec) : parent_class() { (*this) = vec; }

	template<class U, class alt_LD>
	Cube<T, row, col, depth, lib, LD>& operator =(const Cube<U, row, col, depth, lib, alt_LD>& v) {
		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}
	template<class alt_LD>
	Cube<T, row, col, depth, lib, LD>& operator =(const Cube<T, row, col, depth, lib, alt_LD>& v) {
		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}

		  Matrix<T, row, col, lib, LD> operator [] (int index) {
		return Matrix<T, row, col, lib, LD>(& this->array[index * row * col]);
	}
	const Matrix<T, row, col, lib, LD> operator [] (int index) const {
			return Matrix<T, row, col, lib, LD>(& this->array[index * row * col]);
	}

};
}

#endif /* BC_TENSOR_CUBE_H_ */
