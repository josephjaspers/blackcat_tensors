/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include "BC_Tensor_Base/TensorBase.h"

namespace BC {

template<class T, class Mathlib>
class Scalar : public TensorBase<Scalar<T, Mathlib>> {
	struct DISABLE;
	using parent_class = TensorBase<Scalar<T, Mathlib>>;

public:
	__BCinline__ static constexpr int DIMS() { return 0; }
	operator _scalar<T>() const { _scalar<T> value = 0; Mathlib::DeviceToHost(&value, this->data().getIterator(), 1); return value; }

	using parent_class::operator=;
	using parent_class::operator();

	Scalar() : parent_class(std::vector<int>{1}) {}
	Scalar(const Scalar&& t) : parent_class(t) 		{}
	Scalar(		 Scalar&& t) : parent_class(t) 		{}
	Scalar(const Scalar&  t) : parent_class(t) 		{}

	template<class U> 		  Scalar(const Scalar<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Scalar(	   Scalar<U, Mathlib>&& t) : parent_class(t) {}
	template<class... params> Scalar(const params&... p) : parent_class( p...) {}

	Scalar& operator =(const Scalar&  t) { return parent_class::operator=(t); }
	Scalar& operator =(const Scalar&& t) { return parent_class::operator=(t); }
	Scalar& operator =(	     Scalar&& t) { return parent_class::operator=(t); }
	template<class U>
	Scalar& operator =(const Scalar<U, Mathlib>& t) { return parent_class::operator=(t); }
	Scalar& operator =(_scalar<T> scalar) { Mathlib::HostToDevice(this->data().getIterator(), &scalar, 1); return *this; }


	Scalar(_scalar<T> value) : parent_class(std::vector<int>{1}) {
		Mathlib::HostToDevice((_scalar<T>*)this->data(), &value, 1);
	}

	template<class U>	//Constructor that automatically cast primitive types to the scalar_type, IE if scalar<float> but user supplies a double
	Scalar(std::conditional_t<MTF::isPrimitive<U>::conditional && MTF::isPrimitive<T>::conditional, U, DISABLE> value) : parent_class(std::vector<int>{1}) {
		Mathlib::HostToDevice((_scalar<T>*)this->data(), &(T)value, 1);
	}


};


}

#endif /* SCALAR_H_ */
