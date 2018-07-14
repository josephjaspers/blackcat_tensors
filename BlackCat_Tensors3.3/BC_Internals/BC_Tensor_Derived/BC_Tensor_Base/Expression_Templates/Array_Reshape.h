
/*
 * Array_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_RESHAPE_H_
#define TENSOR_RESHAPE_H_

#include "Array_Base.h"

namespace BC {
namespace internal {
/*
 * Accepts an a tensor_core type wrapped in the new_tensor
 *
 * IE if you have a Vector<Array<Vector<float, ml>, ml>  and wish to Reshape to a Matrix
 * The resulting reshape will be-- Matrix<Array_Reshape<Matrix<Array<Vector<float, ml>,ml>>>>
 *
 * This is somewhat awkward and atypical of the other Array traits, but it is essential to be able to pass
 * the constexpr int DIMS in some form. The choice to utilize this method opposed to expanding the number of template arguments
 * was to ensure consistency across the determiners.h which are imperative to the template-metaprogramming.
 */

template<int dimension>
struct Array_Reshape {

	template<class PARENT>
	struct implementation : Tensor_Array_Base<implementation<PARENT>, dimension>{

	using scalar = _scalar<PARENT>;

	__BCinline__ static constexpr int DIMS() { return dimension; };
	__BCinline__ static constexpr int ITERATOR() { return dimension; }

	static_assert(PARENT::ITERATOR() == 0 || PARENT::ITERATOR() == DIMS(), "RESHAPE IS NOT SUPPORTED ON NON-CONTINUOUS TENSORS");

	operator const PARENT() const	{ return parent; }

	PARENT parent;
	scalar* array;
	Shape<dimension> shape;

	template<class... integers>
	implementation(const scalar* array_, PARENT parent, integers... ints) : array(const_cast<scalar*>(array_)), parent(parent), shape(ints...) {}
	__BCinline__ const auto inner_shape() const 	{ return shape.inner_shape(); }
	__BCinline__ const auto outer_shape() const 	{ return shape.outer_shape(); }

	__BCinline__ const auto memptr() const { return array; }
	__BCinline__	   auto memptr()   	   { return array; }

	};
};
}
}

#endif /* TENSOR_RESHAPE_H_ */
