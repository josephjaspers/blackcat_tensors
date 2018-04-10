/*
 *
 * BC_Tensor_Primary_Scalar.h
 *  Created on: Nov 27, 2017
 *      Author: joseph
 *
 */

#ifndef BC_TENSOR_PRIMARY_SCALAR_H_
#define BC_TENSOR_PRIMARY_SCALAR_H_

#include "BC_Tensor_Super_Jack.h"

template<class T, class ml, bool isID_Scalar = false>
class Scalar : public Tensor_Jack<T, ml, 1> {
public:
	using functor_type = Scalar<T, ml, isID_Scalar>;
	Scalar<T, ml, isID_Scalar>(T* scalar) {
		this->array = scalar;
	}

	Scalar<T, ml, isID_Scalar>() {
		ml::initialize(this->array, this->size());
	}

	const T& getData() const {
		return *(this->array);
	}
	T& getData() {
		return *(this->array);
	}

	Scalar<T, ml,isID_Scalar>& operator =(T value) {
		ml::set(this->array, value);
		return *this;
	}

	template<class U>
	Scalar<T, ml,isID_Scalar>& operator =(const Scalar<U, ml>& scal) {
		ml::set(this->array, scal.array);
		return *this;
	}
};

//stack scalar
template<class T, class ml>
class Scalar<T, ml, false> : public Tensor_Jack<Scalar<T, ml, false>, ml, 1> {
public:
	using functor_type = Scalar<T, ml, false>;

	T value;

	Scalar<T, ml, false>(T scalar) :
			value(scalar) {
	}

	const T& getData() const {
		return value;
	}
	T& getData() {
		return value;
	}

	Scalar<T, ml, false>& operator =(T value) {
		throw std::invalid_argument("stack_scalar is R value only ");
	}

	template<class U, class isID>
	Scalar<T, ml>& operator =(const Scalar<U, ml>& scal) {
		throw std::invalid_argument("stack_scalar is R value only ");
	}
};

#endif /* BC_TENSOR_PRIMARY_SCALAR_H_ */
