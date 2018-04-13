/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_

#include <vector>
#include "BC_Tensor_Base/TensorBase.h"
class VOID_;

namespace BC {

template<class T, class Mathlib>
class Scalar : public TensorBase<Scalar<T, Mathlib>> {

	using parent_class = TensorBase<Scalar<T, Mathlib>>;

public:

	__BCinline__ static constexpr int DIMS() { return 0; }

	using parent_class::operator=;
	using primitive_scalar = std::conditional_t<MTF::isPrimitive<T>::conditional, T, VOID_>();

	Scalar() : parent_class(std::vector<int>{1}) {}
	Scalar(const Scalar&& t) : parent_class(t) {}
	Scalar(		 Scalar&& t) : parent_class(t) {}
	Scalar(const Scalar&  t) : parent_class(t) {}

	template<class U> Scalar(const Scalar<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> Scalar(	   Scalar<U, Mathlib>&& t) : parent_class(t) {}


	Scalar(_scalar<T> val) : parent_class(std::vector<int>{1}) {
		 Mathlib::HostToDevice(this->data().getIterator(), &val, 1);
	}

	Scalar& operator =(const Scalar&  t) { return parent_class::operator=(t); }
	Scalar& operator =(const Scalar&& t) { return parent_class::operator=(t); }
	Scalar& operator =(	     Scalar&& t) { return parent_class::operator=(t); }
	template<class U>
	Scalar& operator =(const Scalar<U, Mathlib>& t) { return parent_class::operator=(t); }
	Scalar& operator =(_scalar<T> scalar) { Mathlib::HostToDevice(this->data().getIterator(), &scalar, 1); return *this; }

	operator _scalar<T>() const {
		_scalar<T> value = 0;
		Mathlib::DeviceToHost(&value, this->data().getIterator(), 1);
		return value;
	}

private:

	template<class U> friend class TensorBase;
	template<class U> friend class Tensor_Operations;
	template<class... params> Scalar(const params&... p) : parent_class( p...) {}

};


}

#endif /* SCALAR_H_ */
