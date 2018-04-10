
/*
 * BC_Tensor_Super_King.h
 *
 *  Created on: Nov 20, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_QUEEN_H_
#define BC_TENSOR_SUPER_QUEEN_H_

#include "BC_Tensor_Super_King.h"
#include "BC_Tensor_Super_Shape_Identity.h"

template<class T, class ml, bool ID>
class Scalar;

template<class T, class ml, int ... dimensions>
class Tensor_Queen : public Tensor_King<T, binary_expression, dimensions...> {

	/*
	 * Tensor_Ace will either generate an array pointer of type T or it will
	 * be of a specialized class for Expression classes
	 *
	 * Tensor_King defines the pointwise mathematical functions between other T
	 * Tensors. Expressions inherit from Tensor_King as well as primary Tensors
	 */

public:
	using functor_type = typename Tensor_King<T, binary_expression, dimensions...>::functor_type;
	using shape_identity = typename BC_Identity<T, ml, dimensions...>::type;

	//Operation by Tensor
	template<class U>
	binary_expression<BC::add, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...> operator +(const Tensor_Queen<U, ml, dimensions...>& rv) {
		return binary_expression<BC::add, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...>(this->data(), rv.data());
	}

	template<class U>
	binary_expression<BC::sub, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...> operator -(const Tensor_Queen<U, ml, dimensions...>& rv) {
		return binary_expression<BC::sub, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...>(this->data(), rv.data());
	}

	template<class U>
	binary_expression<BC::mul, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...> operator %(const Tensor_Queen<U, ml, dimensions...>& rv) {
		return binary_expression<BC::mul, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...>(this->data(), rv.data());
	}

	template<class U>
	binary_expression<BC::div, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...> operator /(const Tensor_Queen<U, ml, dimensions...>& rv) {
		return binary_expression<BC::div, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...>(this->data(), rv.data());
	}

	//Operation by Scalr
	template<class U, bool id>
	binary_expression<BC::add, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...> operator +(const Scalar<U, ml, id>& rv) {
		return binary_expression<BC::add, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...>(this->data(), rv);
	}

	template<class U, bool id>
	binary_expression<BC::sub, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...> operator -(const Scalar<U, ml, id>& rv) {
		return binary_expression<BC::sub, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...>(this->data(), rv);
	}

	template<class U, bool id>
	binary_expression<BC::mul, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...> operator %(const Scalar<U, ml, id>& rv) {
		return binary_expression<BC::mul, ml, functor_type, typename Scalar<U, ml, id>::functor_type, dimensions...>(this->data(), rv);
	}

	template<class U, bool id>
	binary_expression<BC::div, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...> operator /(const Scalar<U, ml, id>& rv) {
		return binary_expression<BC::div, ml, functor_type, typename Tensor_Queen<U, ml, dimensions...>::functor_type, dimensions...>(this->data(), rv);
	}

};

#include "BC_Expression_Binary.h"

#endif /* BC_TENSOR_SUPER_QUEEN_H_ */
