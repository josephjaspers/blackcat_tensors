/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_

#include <vector>
#include "BC_Tensor_Base/Tensor_Base.h"
class VOID_;

namespace BC {

template<class T, class Mathlib>
class Scalar : public Tensor_Base<Scalar<T, Mathlib>> {

	using parent_class = Tensor_Base<Scalar<T, Mathlib>>;

public:

	__BCinline__ static constexpr int DIMS() { return 0; }
	using parent_class::operator=;

	Scalar() : parent_class(Shape<0>()) {}
	Scalar(		 Scalar&& t) : parent_class(t) {}
	Scalar(const Scalar&  t) : parent_class(t) {}

	template<class U> Scalar(const Scalar<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> Scalar(	   Scalar<U, Mathlib>&& t) : parent_class(t) {}

	Scalar(_scalar<T> val) : parent_class(Shape<0>()) { Mathlib::HostToDevice(this->internal().memptr(), &val, 1); }
	template<class... params> Scalar(const params&... p) : parent_class( p...) {}

	template<class U>
	Scalar& operator =(const Scalar<U, Mathlib>& t) { return parent_class::operator=(t); }
	Scalar& operator =(const Scalar&  t) { return parent_class::operator=(t); }
	Scalar& operator =(	     Scalar&& t) { return parent_class::operator=(t); }
	Scalar& operator =(_scalar<T> scalar) { Mathlib::HostToDevice(this->internal().memptr(), &scalar, 1); return *this; }

	operator _scalar<T>() const {
		_scalar<T> value = 0;
		Mathlib::DeviceToHost(&value, this->internal().memptr(), 1);
		return value;
	}
};
}

#endif /* SCALAR_H_ */
