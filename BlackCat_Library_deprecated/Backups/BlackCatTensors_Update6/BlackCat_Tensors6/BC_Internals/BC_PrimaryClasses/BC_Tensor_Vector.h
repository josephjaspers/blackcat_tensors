/*
 * BC_Tensor_Base2.h
 *
 *  Created on: Dec 12, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_VECTOR_H_
#define BC_TENSOR_VECTOR_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
#include "BC_Tensor_Scalar.h"
namespace BC {
	template<class, int, class, class >
	class RowVector;

	template<
		class T,
		int   row,
		class lib, 						//default = CPU,
		class LD 						// default = typename DEFAULT_LD<Inner_Shape<row>>::type
	>
	class Vector : public Tensor_Mathematics_Head<T, Vector<T, row, lib, LD>, lib, Static_Inner_Shape<row>, typename DEFAULT_LD<Static_Inner_Shape<row>>::type> {

	public:

		using functor_type = typename Tensor_FunctorType<T>::type;
		using parent_class = Tensor_Mathematics_Head<T, Vector<T, row, lib, LD>, lib, Static_Inner_Shape<row>, typename DEFAULT_LD<Static_Inner_Shape<row>>::type>;
		using grandparent_class = typename parent_class::grandparent_class;
		using this_type = Vector<T, row, lib, LD>;

		using parent_class::operator =;
		using parent_class::parent_class;
		static constexpr Tensor_Shape RANK = VECTOR;



		Vector(int rows) : parent_class({rows}) {};
		template<class U, class alt_LD> Vector(const Vector<U, row, lib, alt_LD>&  vec) : parent_class() { (*this) = vec; }
		template<class U, class alt_LD> Vector(	 	 Vector<U, row, lib, alt_LD>&& vec) : parent_class() { (*this) = vec; }

		template<class U, class alt_LD>
		Vector<T, row, lib, LD>& operator =(const Vector<U, row, lib, alt_LD>& v) {

			this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
					lib::copy(this->data(), v.data(), this->size()) :
					lib::copy_single_thread(this->data(), v.data(), this->size());

			return *this;
		}
		template<class alt_LD>
		Vector<T, row, lib, LD>& operator =(const typename BC_MTF::IF_ELSE<grandparent_class::ASSIGNABLE, Vector<T, row, lib, alt_LD>&, VOID_CLASS> v) {

			this->size() > OPENMP_SINGLE_THREAD_THRESHHOLD ?
					lib::copy(this->data(), v.data(), this->size()) :
					lib::copy_single_thread(this->data(), v.data(), this->size());

			return *this;
		}

		const RowVector<T, row, lib, LD> t() const {
			return RowVector<T, row, lib, LD>(this->data());
		}

		template<int sub_row>
		const Scalar<T, lib> operator [](int index) const {
			return Scalar<T, lib>(&(this->array[index]));
		}
			  Scalar<T, lib> operator [](int index) {
			return Scalar<T, lib>(&(this->array[index]));
		}
	};
}
#include "BC_Tensor_RowVector.h"

#endif /* BC_TENSOR_VECTOR_H_ */
