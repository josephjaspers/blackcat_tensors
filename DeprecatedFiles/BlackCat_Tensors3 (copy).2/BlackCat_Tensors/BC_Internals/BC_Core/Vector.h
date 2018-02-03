/*
 * Vector.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "Scalar.h"
#include "Tensor_Base.h"


namespace BC {
template<class T, class Mathlib>
class Vector : public Tensor_Base<T, Vector<T, Mathlib>, Mathlib> {

	using parent_class = Tensor_Base<T, Vector<T, Mathlib>, Mathlib>;
	using _int = typename parent_class::subAccess_int;
	using __int = typename parent_class::force_evaluation_int;
	template<class,class> friend class Vector;

public:

	static constexpr int RANK() { return 1; }
	using parent_class::operator=;
	using parent_class::parent_class;

	Vector(int dim) : parent_class({dim}) {} Vector(Vector<T, Mathlib>&& vec) : parent_class(vec.expression_packet(), vec.data()) {}

	template<class U> Vector(const Vector<U, Mathlib>& vec) : parent_class(vec.expression_packet()) { Mathlib::copy(this->data(), vec.data(), this->size()); }
	 Vector(const Vector<T, Mathlib>& vec) : parent_class(vec.expression_packet()) { Mathlib::copy(this->data(), vec.data(), this->size()); }


	Scalar<T, Mathlib> operator[] (_int i) {
		return (Scalar<T, Mathlib>(&this->array[i]));
	}

	const Scalar<T, Mathlib> operator[] (_int i) const {
		return Scalar<T, Mathlib>(&this->array[i]);
	}

	const Vector<unary_expression_transpose<typename MTF::determine_scalar<T>::type, Vector<T, Mathlib>>, Mathlib> t() const {
		return Vector<unary_expression_transpose<typename MTF::determine_scalar<T>::type, Vector<T, Mathlib>>, Mathlib>(this->transpose_packet(), *this);
	}
	auto operator [] (__int i) const {
		return this->data()[i];
	}

	Vector<T, Mathlib>& operator = (const Vector<T, Mathlib>& vec) {
		this->assert_same_size(vec);
		Mathlib::copy(this->data(), vec.data(), this->size());
		return this->asBase();
	}

	template<class U>
	Vector<T, Mathlib>& operator = (const Vector<U, Mathlib>& vec) {
		this->assert_same_size(vec);
		Mathlib::copy(this->data(), vec.data(), this->size());
		return this->asBase();
	}

};

} //End Namespace BC

#endif /* VECTOR_H_ */
