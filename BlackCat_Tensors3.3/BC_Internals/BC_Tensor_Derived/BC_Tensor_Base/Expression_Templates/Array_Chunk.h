
/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CHUNK_H_
#define TENSOR_CHUNK_H_

#include "Array_Base.h"

namespace BC {
namespace internal {
template<int dimension>
struct Array_Chunk  {

	template<class PARENT>
	struct implementation : Tensor_Array_Base<implementation<PARENT>,dimension> {

		static_assert(PARENT::ITERATOR() == 0 || dimension <= PARENT::DIMS(), "TENSOR-CHUNK'S DIMENSIONS MUST BE LESS OR EQUAL TO PARENT'S DIMENSIONS");

		using scalar = _scalar<PARENT>;

		__BCinline__ static constexpr int DIMS() 	{ return dimension; };
		__BCinline__ static constexpr int ITERATOR() { return dimension; }

		operator const PARENT() const	{ return parent; }

		PARENT parent;
		scalar* array;
		Shape<DIMS()> shape;

		template<class... integers>
		implementation(const scalar* array_, PARENT parent, integers... ints) : array(const_cast<scalar*>(array_)), parent(parent), shape(ints...) {
		}
		__BCinline__ const auto size()		 const      { return shape.size(); }
		__BCinline__ const auto inner_shape() const 	{ return shape.inner_shape(); }
		__BCinline__ const auto outer_shape() const 	{ return parent.outer_shape(); }

		__BCinline__ const auto memptr() const { return array; }
		__BCinline__	   auto memptr()   	   { return array; }
	};
	};
}
}

#endif /* TENSOR_RESHAPE_H_ */
