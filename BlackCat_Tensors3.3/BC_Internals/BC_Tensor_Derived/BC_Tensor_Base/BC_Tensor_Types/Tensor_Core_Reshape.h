
/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_RESHAPE_H_
#define TENSOR_RESHAPE_H_

#include "BlackCat_Tensor_Core_Base.h"

namespace BC {

template<class PARENT, int NEW_DIM>
struct Tensor_Reshape : Tensor_Core_Base<Tensor_Reshape<PARENT, NEW_DIM>, NEW_DIM> {

	using Mathlib = typename  PARENT::Mathlib;

	__BCinline__ static constexpr int DIMS() { return NEW_DIM; };
	__BCinline__ static constexpr int CONTINUOUS() { return 0; }
	static_assert(PARENT::CONTINUOUS() == 0, "Tensor_Reshape may only reshape continuous tensors, you may attempt to copy instead");

	static_assert(NEW_DIM > -1, "DIMENSIONALITY OF TENSOR MUST BE >= 0");

	PARENT parent;
	int* is = Mathlib::unified_initialize(is, DIMS());
	int* os = Mathlib::unified_initialize(os, DIMS());

	operator const PARENT	   () const	{ return parent; }

	template<int dim> void init() {/*DONT DELETE I HELP COMPILE*/ }
	template<int dim, class... integers>
	void init(int front, integers... ints) {
		is[dim] = front;
		if (dim > 0)
			os[dim] = front * os[dim - 1];
		else
			os[0] = front;

		if (dim != DIMS() - 1) {
			init<(dim + 1 < DIMS() ? dim + 1 : dim)>(ints...);
		}
	}

	template<class... integers>
	Tensor_Reshape(PARENT parent, integers... ints) : parent(parent) {
		static_assert(sizeof...(integers) == DIMS(), "DIMENSIONALITY OF RESHAPE MUST EQUAL THE PARAMETER INTEGERS FOR RESHAPE");
		init<0>(ints...);
		if (this->size() != parent.size()) {
			std::cout << "TENSOR RESHAPE SIZE MUST BE EQUAL TO ITS ORIGINAL SIZE" << std::endl;
			throw std::invalid_argument("INVALID RESHAPE");
		}
	}

	__BCinline__ const auto innerShape() const 	{ return is; }
	__BCinline__ const auto outerShape() const 	{ return os; }

	__BCinline__ const auto getIterator() const { return parent.getIterator(); }
	__BCinline__	   auto getIterator()   	{ return parent.getIterator(); }



	};
}

#endif /* TENSOR_RESHAPE_H_ */
