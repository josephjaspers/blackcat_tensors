/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include <type_traits>
#include "TensorBase.h"

namespace BC {

template<class T, class Mathlib>
class Scalar : public TensorBase<Scalar<T, Mathlib>> {

	using parent_class = TensorBase<Scalar<T, Mathlib>>;
	template<class, class> friend class Vector;

public:
	struct DISABLE;
	__BCinline__ static constexpr int DIMS() { return 0; }

	template<bool var, class a, class b>
	using ifte = std::conditional_t<var, a, b>;
	using _shape = std::vector<int>;
	using parent_class::operator=;
	using parent_class::operator();

	Scalar() : parent_class(std::vector<int>{1}) {}
	Scalar(const Scalar&& t) : parent_class(t) 		{}
	Scalar(		 Scalar&& t) : parent_class(t) 		{}
	Scalar(const Scalar&  t) : parent_class(t) 		{}

	operator _scalar<T>() const { _scalar<T> value = 0; Mathlib::DeviceToHost(&value, this->data().core(), 1); return value; }

	Scalar& operator =(const Scalar&  t) { return parent_class::operator=(t); }
	Scalar& operator =(const Scalar&& t) { return parent_class::operator=(t); }
	Scalar& operator =(	     Scalar&& t) { return parent_class::operator=(t); }
	template<class U>
	Scalar& operator =(const Scalar<U, Mathlib>& t) { return parent_class::operator=(t); }
	Scalar& operator =(_scalar<T> scalar) { Mathlib::HostToDevice(this->data().getIterator(), &scalar, 1); return *this; }

	Scalar(_scalar<T>* param): parent_class(param) {}

	//this how to do constexpr if -esque things inside of an initialization list in a constructor
	struct sendParam 	{ template<class u>  	static auto impl(const u& param) 	{ return param; }};
	struct sendNull 	{ template<class u>  	static auto impl(const u& param) 	{ return _shape(); }};
	struct htd 			{ template<class... u>  static void impl(const u&... param) { Mathlib::HostToDevice(param...); }};
	struct voider 		{ template<class... u>  static void impl(const u&... parma) {}};

	explicit Scalar(const ifte<MTF::isPrimitive<T>::conditional, DISABLE, const T&> value) : parent_class(value) {}
	Scalar(_scalar<T> value) : parent_class(std::vector<int>{1}) {
		Mathlib::HostToDevice((_scalar<T>*)this->data(), &value, 1);
	}

	template<class var1, class var2, class... params>
	explicit Scalar(const var1& v1, const var2& v2, const params&... p) : parent_class(v1, v2, p...) {}

};


}



#endif /* SCALAR_H_ */
