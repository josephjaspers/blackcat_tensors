/*
 * Scalar.h
 *
 *  Created on: Jan 6, 2018
 *      Author: joseph
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include "Tensor_Base.h"

namespace BC {

template<class T, class Mathlib>
class Scalar : public Tensor_Base<T, Scalar<T, Mathlib>, Mathlib>
{

	using parent_class = Tensor_Base<T, Scalar<T, Mathlib>, Mathlib>;
	template<class, class> friend class Vector;


public:
	static constexpr int RANK() { return 0; }
	using parent_class::parent_class;

	template<class U, class V>
	Scalar(const U& u, const V& v) : parent_class(u, v) {}

	template<class U>
	Scalar<T, Mathlib>& operator =(const Scalar<U, Mathlib>& scalar) {
		Mathlib::set_heap(this->data(), scalar.data());
		return *this;
	}

	operator 	   T&() 	  { return *(this->data()); }
	operator const T&() const { return *(this->data()); }


	template<class U>
	Scalar<T, Mathlib>& operator =(U&& scalar) {
		Mathlib::set_stack(this->data(), scalar);
		return *this;
	}

	Scalar(T* param) : parent_class(param) {}
	explicit Scalar(T value) { Mathlib::set_stack(this->array, value); }
};


}



#endif /* SCALAR_H_ */
