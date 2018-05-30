
/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CHUNK_H_
#define TENSOR_CHUNK_H_

#include "Core_Base.h"

namespace BC {

template<int dimension>
struct Tensor_Chunk  {
	template<class PARENT>
	struct implementation : Tensor_Core_Base<implementation<PARENT>,dimension> {

		using scalar = _scalar<PARENT>;
		static_assert(dimension <= PARENT::DIMS(), "TENSOR-CHUNK'S DIMENSIONS MUST BE LESS OR EQUAL TO PARENT'S DIMENSIONS");

		__BCinline__ static constexpr int DIMS() { return dimension; };
		__BCinline__ static constexpr int ITERATOR() { return dimension; }

		operator const PARENT() const	{ return parent; }

		PARENT parent;
		scalar* array;
		Shape<DIMS()> shape;

		template<class... integers> __BCinline__
		implementation(const scalar* array_, PARENT parent, integers... ints) : array(const_cast<scalar*>(array_)), parent(parent), shape(ints...) {
		}
		__BCinline__ const auto size()		 const  { return shape.size(); }
		__BCinline__ const auto innerShape() const 	{ return shape.is(); }
		__BCinline__ const auto outerShape() const 	{ return parent.outerShape(); }

		__BCinline__ const auto getIterator() const { return array; }
		__BCinline__	   auto getIterator()   	{ return array; }
	};
	};
}

#endif /* TENSOR_RESHAPE_H_ */
