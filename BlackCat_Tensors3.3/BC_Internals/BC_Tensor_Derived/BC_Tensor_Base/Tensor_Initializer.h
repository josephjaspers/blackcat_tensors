/*
 * Tensor_Initializer.h
 *
 *  Created on: Mar 3, 2018
 *      Author: joseph
 */

#ifndef TENSOR_INITIALIZER_H_
#define TENSOR_INITIALIZER_H_

#include "Determiners.h"
#include "MetaTemplateFunctions.h"

#include "Implementation_Core/Tensor_Core.h"
#include "Implementation_Core/Tensor_Core_Scalar.h"
#include "Implementation_Core/Tensor_Core_Slice.h"
#include "Implementation_Core/Tensor_Core_Reshape.h"
#include "Implementation_Core/Tensor_Core_RowVector.h"

namespace BC {

//-------------------------------------SPECIALIZATION FOR EXPRESSION TENSORS OR TENSORS OF NON_OWNERSHIP/CREATION-------------------------------------//
template<class derived, class expression_tensor = void>
class TensorInitializer {

	using self 			= TensorInitializer<derived>;

	using functor_type 	= _functor<derived>;
	using Mathlib 		= _mathlib<derived>;
	using scal			= _scalar<derived>;

public:

	functor_type black_cat_array;

	TensorInitializer(		 derived&& tensor) : black_cat_array(tensor.black_cat_array){}
	TensorInitializer(const  derived&  tensor) : black_cat_array(tensor.black_cat_array){}

	template<class... params>
	explicit TensorInitializer(const  params&... p) : black_cat_array(p...) {}
	~TensorInitializer() {
		black_cat_array.destroy();
	}
};
//-------------------------------------SPECIALIZATION FOR TENSORS THAT CONTROL / DELETE THEIR ARRAY-------------------------------------//
template<template<class, class> class _tensor, class t, class ml>
class TensorInitializer<_tensor<t, ml>, std::enable_if_t<MTF::isPrimitive<t>::conditional>> {

	using derived = _tensor<t, ml>;
	static constexpr int DIMS = derived::DIMS();

	using self 			= TensorInitializer<derived, std::enable_if_t<MTF::isPrimitive<t>::conditional>>;

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

	TensorInitializer(derived&& tensor) {
		black_cat_array.is = tensor.black_cat_array.is;
		black_cat_array.os = tensor.black_cat_array.os;
		black_cat_array.array = tensor.black_cat_array.array;

		tensor.black_cat_array.is 		= nullptr;
		tensor.black_cat_array.os 		= nullptr;
		tensor.black_cat_array.array 	= nullptr;
	}

	TensorInitializer(const derived& tensor) : black_cat_array(tensor.innerShape()) {
		Mathlib::copy(asBase().data(), tensor.data(), tensor.size());
	}
	TensorInitializer(_shape dimensions): black_cat_array(dimensions) {}

	template<class T>
	using derived_alt = typename MTF::shell_of<derived>::template  type<T, Mathlib>;

	template<class U>
	TensorInitializer(const derived_alt<U>&  tensor)
		: black_cat_array(tensor.innerShape()) {
		Mathlib::copy(this->asBase().data(), tensor.data(), this->asBase().size());
	}

	~TensorInitializer() {
		if (black_cat_array.array)
			Mathlib::destroy (black_cat_array.array);
		if (black_cat_array.is)
			Mathlib::destroy (black_cat_array.is);
		if (black_cat_array.os)
			Mathlib::destroy (black_cat_array.os);
	}
};

}



#endif /* TENSOR_INITIALIZER_H_ */
