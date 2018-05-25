
/*
 * Tensor_Reshape.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CHUNK
#define TENSOR_CHUNK

#include "Core_Base.h"

/*
 * I HAVE NOT BEEN TESTED YET, TEST ME
 * I RETURN AN INNER CHUNK OF A TENSOR
 *
 */

namespace BC {

template<int dimension>
struct Tensor_Chunk  {
	template<class PARENT>
	struct implementation : Core_Base<implementation<PARENT>,dimension> {

		using scalar = _scalar<PARENT>;

		__BCinline__ static constexpr int DIMS() { return dimension; };
		__BCinline__ static constexpr int ITERATOR() { return dimension; }

		operator const PARENT() const	{ return parent; }

		PARENT parent;
		scalar* array;
		Shape<DIMS()> shape;

		template<class... integers>
		implementation(scalar* array_, PARENT parent_, integers... ints) : parent(parent_), array(array_), shape(ints...) {}

		__BCinline__ const auto size()		 const  { return this->os[DIMS() - 1]; }
		__BCinline__ const auto innerShape() const 	{ return shape.is(); }
		__BCinline__ const auto outerShape() const 	{ return parent.outerShape(); }

		__BCinline__ const auto getIterator() const { return array; }
		__BCinline__	   auto getIterator()   	{ return array; }
	};
	};
}

#endif /* TENSOR_RESHAPE_H_ */
