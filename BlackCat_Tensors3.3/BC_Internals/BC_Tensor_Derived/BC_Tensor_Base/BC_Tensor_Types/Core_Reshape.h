
/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_RESHAPE_H_
#define TENSOR_RESHAPE_H_

#include "Core_Base.h"

namespace BC {

template<class PARENT>
struct Tensor_Reshape;

/*
 * Accepts an a tensor_core type wrapped in the new_tensor
 *
 * IE if you have a Vector<Core<Vector<float, ml>, ml>  and wish to Reshape to a Matrix
 * The resulting reshape will be-- Matrix<Tensor_Reshape<Matrix<Core<Vector<float, ml>,ml>>>>
 *
 * This is somewhat awkward and atypical of the other Core traits, but it is essential to be able to pass
 * the constexpr int DIMS in some form. The choice to utilize this method opposed to expanding the number of template arguments
 * was to ensure consistency across the determiners.h which are imperative to the template-metaprogramming.
 */

template<template<class...> class TENSOR, class PARENT, class math_lib>
struct Tensor_Reshape<TENSOR<PARENT, math_lib>> : Core_Base<Tensor_Reshape<TENSOR<PARENT, math_lib>>, TENSOR<PARENT, math_lib>::DIMS()> {

	__BCinline__ static constexpr int PARENT_DIMS() { return PARENT::PARENT_DIMS(); }
	__BCinline__ static constexpr int DIMS() { return class2rank<TENSOR<PARENT, math_lib>>; };
	__BCinline__ static constexpr int CONTINUOUS() { return 0; }

//	static_assert(PARENT::CONTINUOUS() == 0, "Tensor_Reshape may only reshape continuous tensors, you may attempt to copy instead");

	PARENT parent;
	int is[DIMS()];
	int os[DIMS()];

	__BCinline__
	operator const PARENT	   () const	{ return parent; }

	template<int dim, class... integers> __BCinline__
	void init(integers... ints) {

		auto vars = array(ints...);

		is[0] = vars[0];
		os[0] = vars[0];
		for (int i = 1; i < vars.size(); ++i) {
			is[i] = vars[i];
			os[i] = is[i] * os[i - 1];
		}
	}


	template<class... integers> __BCinline__
	Tensor_Reshape(PARENT parent, integers... ints) : parent(parent) {
		static_assert(sizeof...(integers) == DIMS(), "DIMENSIONALITY OF RESHAPE MUST EQUAL THE PARAMETER INTEGERS FOR RESHAPE");
		init<0>(ints...);
	}

	__BCinline__ const auto innerShape() const 	{ return is; }
	__BCinline__ const auto outerShape() const 	{ return os; }

	__BCinline__ const auto getIterator() const { return parent.getIterator(); }
	__BCinline__	   auto getIterator()   	{ return parent.getIterator(); }



	};
}

#endif /* TENSOR_RESHAPE_H_ */
