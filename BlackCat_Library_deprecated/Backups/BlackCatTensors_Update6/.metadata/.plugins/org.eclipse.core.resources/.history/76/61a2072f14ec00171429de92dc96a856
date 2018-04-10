/*
 * BC_Tensor_RowRowVector.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */



#ifndef BC_TENSOR_ROWRowVector_H_
#define BC_TENSOR_ROWRowVector_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
#include "BC_Tensor_Scalar.h"
namespace BC {
template<class, int, int, class, class>
class Matrix;


template<class T, int row, class lib, //default = CPU,
		class LD // default = typename DEFAULT_LD<Inner_Shape<row>>::type
>
class RowVector : public Tensor_Mathematics_Head<T, RowVector<T, row, lib, LD>, lib, Static_Inner_Shape<1, row>, typename DEFAULT_LD<Static_Inner_Shape<1, row>>::type> {

public:


	using functor_type = typename Tensor_FunctorType<T>::type;
	using parent_class = Tensor_Mathematics_Head<T, RowVector<T, row, lib, LD>, lib, Static_Inner_Shape<1, row>, typename DEFAULT_LD<Static_Inner_Shape<1, row>>::type>;
	using grandparent_class = typename parent_class::grandparent_class;
	using this_type = RowVector<T, row, lib, LD>;
	static constexpr Tensor_Shape RANK = VECTOR;

	using parent_class::parent_class;

	template<class U, class alt_LD> RowVector(const RowVector<U, row, lib, alt_LD>&  vec) : parent_class() { (*this) = vec; }
	template<class U, class alt_LD>	RowVector(      RowVector<U, row, lib, alt_LD>&& vec) : parent_class() { (*this) = vec; }

	template<class U, class alt_LD>
	RowVector<T, row, lib, LD>& operator =(const RowVector<U, row, lib, alt_LD>& v) {
		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}
	template<class alt_LD>
	RowVector<T, row, lib, LD>& operator =(const typename BC_MTF::IF_ELSE<grandparent_class::ASSIGNABLE, RowVector<T, row, lib, alt_LD>&, VOID_CLASS> v) {
		this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
			lib::copy(this->data(), v.data(), this->size()):
			lib::copy_single_thread(this->data(), v.data(), this->size());

		return *this;
	}

	const Vector<T, row, lib, LD> t() const {
		return Vector<T, row, lib, LD>(this->data());
	}


	template<int sub_row>
	const Scalar<T, lib> operator [] (int index) const {
		return Scalar<T, lib>(&(this->array[index]));
	}
		  Scalar<T, lib> operator [] (int index) {
		return Scalar<T, lib>(&(this->array[index]));
	}

};
}

#endif /* BC_TENSOR_ROWRowVector_H_ */
