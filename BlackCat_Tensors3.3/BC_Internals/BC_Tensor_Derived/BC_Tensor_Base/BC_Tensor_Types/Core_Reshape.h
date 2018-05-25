
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

template<int dimension>
struct Tensor_Reshape {

	template<class PARENT>
	struct implementation : Tensor_Core_Base<implementation<PARENT>, dimension>{

	using scalar = _scalar<PARENT>;

	__BCinline__ static constexpr int DIMS() { return dimension; };
	__BCinline__ static constexpr int ITERATOR() { return dimension; }

	static_assert(PARENT::ITERATOR() == 0, "RESHAPE IS NOT SUPPORTED ON NON-CONTINUOUS TENSORS");

	operator const PARENT() const	{ return parent; }

	PARENT parent;
	scalar* array;
	Shape<dimension> shape;

	template<class ary, class... integers> __BCinline__
	implementation(ary array_, PARENT parent, integers... ints) : array(array_), parent(parent), shape(ints...) {
		static_assert(sizeof...(integers) == DIMS(), "DIMENSIONALITY OF RESHAPE MUST EQUAL THE PARAMETER INTEGERS FOR RESHAPE");
	}

	__BCinline__ const auto innerShape() const 	{ return shape.is(); }
	__BCinline__ const auto outerShape() const 	{ return shape.os(); }

	__BCinline__ const auto getIterator() const { return array; }
	__BCinline__	   auto getIterator()   	{ return array; }

	};
};
}

#endif /* TENSOR_RESHAPE_H_ */
