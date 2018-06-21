/*
 * Tensor_Initializer.h
 *
 *  Created on: Mar 3, 2018
 *      Author: joseph
 */

#ifndef TENSOR_INITIALIZER_H_
#define TENSOR_INITIALIZER_H_

#include "BC_Internal_Types/Array.h"
#include "BC_Internal_Types/Array_Scalar.h"
#include "BC_Internal_Types/Array_Slice.h"
#include "BC_Internal_Types/Array_RowVector.h"
#include "BC_Internal_Types/Array_Chunk.h"
#include "BC_Internal_Types/Array_Reshape.h"

namespace BC {
namespace Base {
//-------------------------------------SPECIALIZATION FOR EXPRESSION TENSORS OR TENSORS OF NON_OWNERSHIP/CREATION-------------------------------------//
template<class derived, class expression_tensor = void>
class Tensor_Initializer :  public _functor<derived> {

	using self 			= Tensor_Initializer<derived>;

	using array 		= _functor<derived>;
	using mathlib_t 	= _mathlib<derived>;
	using scalar		= _scalar<derived>;

public:

	 const array& internal() const { return static_cast<const array&>(*this); }
	 array& internal() 	  { return static_cast<		 array&>(*this); }

	Tensor_Initializer(		  derived&& tensor) : array(tensor.internal()){}
	Tensor_Initializer(const  derived&  tensor) : array(tensor.internal()){}

	template<class... params>
	explicit Tensor_Initializer(const  params&... p) : array(p...) {}

	~Tensor_Initializer() {
		internal().destroy();
	}
};
//-------------------------------------SPECIALIZATION FOR TENSORS THAT CONTROL / DELETE THEIR ARRAY-------------------------------------//


template<template<class, class> class _tensor, class t, class ml>
struct Tensor_Initializer<_tensor<t, ml>, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>> : public  _functor<_tensor<t, ml>> {

	using derived = _tensor<t, ml>;
	using self 		= Tensor_Initializer<derived, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>>;

	using functor_type 	= _functor<derived>;
	using mathlib_t 	= _mathlib<derived>;
	using scalar		= _scalar<derived>;

//	functor_type array_core;

	using array_core = _functor<derived>;


private:

	auto& as_derived() 			   { return static_cast<	  derived&>(*this); }
	const auto& as_derived() const { return static_cast<const derived&>(*this); }

public:

	 const functor_type& internal() const { return static_cast<const _functor<derived>&>(*this); }
	  	   functor_type& internal() 	  { return static_cast<		 _functor<derived>&>(*this); }

	Tensor_Initializer(derived&& tensor) : array_core(tensor.internal()) { tensor.internal().array 	= nullptr; }

	Tensor_Initializer(const derived& tensor) : array_core(tensor.inner_shape()) {
		this->as_derived() = tensor;
	}
	template<class T>
	Tensor_Initializer(T dimensions): array_core(dimensions) {}

	template<class T> using derived_alt = typename MTF::shell_of<derived>::template  type<T, mathlib_t>;

	template<class U>
	Tensor_Initializer(const derived_alt<U>&  tensor)
		: array_core(tensor.inner_shape()) {
		this->as_derived() = tensor;
	}

	~Tensor_Initializer() {
		internal().destroy();
	}
};
}
}



#endif /* TENSOR_INITIALIZER_H_ */
