/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include "TensorBase.h"

namespace BC {

template<class T, class Mathlib>
class Scalar : public TensorBase<T, Scalar<T, Mathlib>, Mathlib, Rank<0>> {

	using parent_class = TensorBase<T, Scalar<T, Mathlib>, Mathlib, Rank<0>>;
	template<class, class> friend class Vector;

public:
	static constexpr int RANK() { return 0; }

	using parent_class::operator=;
	using param_tc = typename parent_class::param_tc;

	Scalar() {}
	Scalar(const Scalar&& t) : parent_class(t) 		{}
	Scalar(		 Scalar&& t) : parent_class(t) 		{}
	Scalar(const Scalar&  t) : parent_class(t) 		{}
	template<class... params> Scalar(param_tc sh, const params&... p) : parent_class(sh, p...) {}

	Scalar& operator =(const Scalar&  t) { return parent_class::operator=(t); }
	Scalar& operator =(const Scalar&& t) { return parent_class::operator=(t); }
	Scalar& operator =(	     Scalar&& t) { return parent_class::operator=(t); }
	template<class U>
	Scalar& operator =(const Scalar<U, Mathlib>& t) { return parent_class::operator=(t); }
	Scalar& operator =(T scalar) { Mathlib::DeviceToHost(this->data(), &scalar, 1); return *this; }

	using _shape = std::vector<int>;
	Scalar(T* param) {}
	Scalar(T value) {
		Mathlib::HostToDevice(this->data().ary(), &value, 1);
	}
};


}



#endif /* SCALAR_H_ */
