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
#include "BC_Tensor_Types/Core_Reshape.h"
#include "BC_Tensor_Types/Core_RowVector.h"
#include "BC_Tensor_Types/Core_Chunk.h"

namespace BC {

//-------------------------------------SPECIALIZATION FOR EXPRESSION TENSORS OR TENSORS OF NON_OWNERSHIP/CREATION-------------------------------------//
template<class derived, class expression_tensor = void>
class Tensor_Initializer {

	using self 			= Tensor_Initializer<derived>;

	using functor_type 	= _functor<derived>;
	using Mathlib 		= _mathlib<derived>;
	using scal			= _scalar<derived>;

public:

	functor_type black_cat_array;

	Tensor_Initializer(		 derived&& tensor) : black_cat_array(tensor.black_cat_array){}
	Tensor_Initializer(const  derived&  tensor) : black_cat_array(tensor.black_cat_array){}

	template<class... params>
	explicit Tensor_Initializer(const  params&... p) : black_cat_array(p...) {}
//
	~Tensor_Initializer() {
		black_cat_array.destroy();
	}
};
//-------------------------------------SPECIALIZATION FOR TENSORS THAT CONTROL / DELETE THEIR ARRAY-------------------------------------//


template<template<class, class> class _tensor, class t, class ml>
class Tensor_Initializer<_tensor<t, ml>, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>> {

	using derived = _tensor<t, ml>;
	using self 			= Tensor_Initializer<derived, std::enable_if_t<!std::is_base_of<BC_Type,t>::value>>;

	using functor_type 	= _functor<derived>;
	using Mathlib 		= _mathlib<derived>;
	using scal			= _scalar<derived>;
	using _shape 		= std::vector<int>;

protected:
	functor_type black_cat_array;
private:
	auto& asBase() 			   { return static_cast<	  derived&>(*this); }
	const auto& asBase() const { return static_cast<const derived&>(*this); }

public:

	Tensor_Initializer(derived&& tensor) : black_cat_array() {
		black_cat_array.is = tensor.black_cat_array.is;
		black_cat_array.os = tensor.black_cat_array.os;
		black_cat_array.array = tensor.black_cat_array.array;

		tensor.black_cat_array.is 		= nullptr;
		tensor.black_cat_array.os 		= nullptr;
		tensor.black_cat_array.array 	= nullptr;
	}

	Tensor_Initializer(const derived& tensor) : black_cat_array(tensor.innerShape()) {
		Mathlib::copy(asBase().data(), tensor.data(), tensor.size());
	}
	template<class T>
	Tensor_Initializer(T dimensions): black_cat_array(dimensions) {}

	template<class T> using derived_alt = typename MTF::shell_of<derived>::template  type<T, Mathlib>;

	template<class U>
	Tensor_Initializer(const derived_alt<U>&  tensor)
		: black_cat_array(tensor.innerShape()) {
		this->asBase() = tensor;
	}

	~Tensor_Initializer() {
		black_cat_array.destroy();
	}
};

}



#endif /* TENSOR_INITIALIZER_H_ */
