/*
 * BC_Accessor_Scalar.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef BC_ACCESSOR_SCALAR_H_
#define BC_ACCESSOR_SCALAR_H_


#include "BC_Accessor.h"

/*
 * This class is currently not used -- it is an alternative
 * to utilzing binary_pointwise_scalar_L/R
 */

template<class T, class ml>
struct accessor_scalar : accessor<T> {
	 T scalar;

	 accessor_scalar(T& scal) : scalar(scal) {};

	 T& operator [] (int index) { return scalar; }
	 const T& operator [] (int index) const { return scalar;}
};


#endif /* BC_ACCESSOR_SCALAR_H_ */






//Alternate method for handling Scalars
//	template<class U>
//	Tensor_King<binary_expression<array_type, BC::add, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...> operator +(const Tensor_King<U, ml, 1>& rv) {
//		return Tensor_King<binary_expression<array_type, BC::add, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...>(this->data(), rv.data());
//	}
//
//	template<class U>
//	Tensor_King<binary_expression<array_type, BC::sub, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...> operator - (const Tensor_King<U, ml, 1>& rv) {
//		return Tensor_King<binary_expression<array_type, BC::sub, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...>(this->data(), rv.data());
//	}
//
//	template<class U>
//	Tensor_King<binary_expression<array_type, BC::div, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...> operator /(const Tensor_King<U, ml, 1>& rv) {
//		return Tensor_King<binary_expression<array_type, BC::div, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...>(this->data(), rv.data());
//	}
//
//	template<class U>
//	Tensor_King<binary_expression<array_type, BC::mul, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...> operator %(const Tensor_King<U, ml, 1>& rv) {
//		return Tensor_King<binary_expression<array_type, BC::mul, functor_type, typename Tensor_King<U, ml, 1>::functor_type>, ml, dimensions...>(this->data(), rv.data());
//	}

//template<class T, class ml>
//class Scalar : public Tensor_Queen<accessor_scalar<T, ml>, ml, 1> {
//
//	template<class, class, int...>
//	friend class Vector;
//public:
//
//	Scalar<T, ml>(T& scalar) : Tensor_Queen<accessor_scalar<T, ml>, ml, 1>(scalar) {}
//
//
//	Scalar<T, ml>& operator = (T value) { ml::set(this->data(), value); return * this;}
//
//	template<typename U>
//	Scalar<T, ml>& operator = (const Scalar<U, ml>& scalar) { ml::set(this->data(), scalar.data()); return *this; }
//};
