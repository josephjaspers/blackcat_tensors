/*
 * Tensor_Initializer.h
 *
 *  Created on: Mar 3, 2018
 *      Author: joseph
 */

#ifndef TENSOR_INITIALIZER_H_
#define TENSOR_INITIALIZER_H_

#include "BC_Tensor_Types/Core.h"
#include "BC_Tensor_Types/Core_Scalar.h"
#include "BC_Tensor_Types/Core_Slice.h"
#include "BC_Tensor_Types/Core_RowVector.h"
#include "BC_Tensor_Types/Core_Chunk.h"
#include "BC_Tensor_Types/Core_Reshape.h"

namespace BC {
namespace Base {
//-------------------------------------SPECIALIZATION FOR EXPRESSION TENSORS OR TENSORS OF NON_OWNERSHIP/CREATION-------------------------------------//
template<class derived, class expression_tensor = void>
class Tensor_Initializer {

	using self 			= Tensor_Initializer<derived>;

	using functor_type 	= _functor<derived>;
	using mathlib_t 		= _mathlib<derived>;
	using scalar			= _scalar<derived>;

public:

	functor_type array_core;

	 const functor_type& data() const { return this->array_core; }
	 	   functor_type& data()		  { return this->array_core; }

	Tensor_Initializer(		 derived&& tensor) : array_core(tensor.array_core){}
	Tensor_Initializer(const  derived&  tensor) : array_core(tensor.array_core){}

	template<class... params>
	explicit Tensor_Initializer(const  params&... p) : array_core(p...) {}

	~Tensor_Initializer() {
		array_core.destroy();
	}
};
//-------------------------------------SPECIALIZATION FOR TENSORS THAT CONTROL / DELETE THEIR ARRAY-------------------------------------//


template<template<class, class> class _tensor, class t, class ml>
struct Tensor_Initializer<_tensor<t, ml>, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>> {

	using derived = _tensor<t, ml>;
	using self 			= Tensor_Initializer<derived, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>>;

	using functor_type 	= _functor<derived>;
	using mathlib_t 	= _mathlib<derived>;
	using scalar		= _scalar<derived>;

	functor_type array_core;

private:

	auto& as_derived() 			   { return static_cast<	  derived&>(*this); }
	const auto& as_derived() const { return static_cast<const derived&>(*this); }

public:

	 const functor_type& data() const { return this->array_core; }
	 	   functor_type& data()		  { return this->array_core; }

	Tensor_Initializer(derived&& tensor) : array_core(tensor.data()) {
//		array_core.shape = tensor.array_core.shape;
//		array_core.array = tensor.array_core.array;
		tensor.array_core.array 	= nullptr;
	}

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
		array_core.destroy();
	}
};
}
}



#endif /* TENSOR_INITIALIZER_H_ */
