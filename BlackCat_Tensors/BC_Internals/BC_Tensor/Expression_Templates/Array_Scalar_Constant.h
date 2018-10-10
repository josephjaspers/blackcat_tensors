/*
 * Expression_Scalar_Constant.h
 *
 *  Created on: Oct 9, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_

#include "Array_Base.h"

namespace BC {
class CPU;
namespace internal {

//identical to Array_Scalar, though the scalar is allocated on the stack opposed to heap
template<class scalar_t_>
struct Scalar_Constant : Shape<0>, Array_Base<Scalar_Constant<scalar_t_>, 0>{

	using scalar_t = scalar_t_;
	using mathlib_t = BC::CPU;

	__BCinline__ static constexpr int ITERATOR() { return 0; }
	__BCinline__ static constexpr int DIMS() 	 { return 0; }

	scalar_t scalar;
	Scalar_Constant(scalar_t scalar_) : scalar(scalar_) {}


	template<class... integers>
	auto operator()  (const integers&...) const{ return scalar; }
	auto operator [] (int i ) { return scalar; }
	const scalar_t* memptr() const { return &scalar; }

	void swap_array(Scalar_Constant&) {}
};

template<class scalar_t>
auto scalar_constant(scalar_t scalar) {
	return Scalar_Constant<scalar_t>(scalar);
}
}
}




#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_SCALAR_CONSTANT_H_ */
