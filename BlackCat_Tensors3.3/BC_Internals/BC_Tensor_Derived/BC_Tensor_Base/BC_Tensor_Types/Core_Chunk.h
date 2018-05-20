
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
		__BCinline__ static constexpr int CONTINUOUS() { return dimension; }

		operator const PARENT() const	{ return parent; }

		PARENT parent;
		scalar* array;

		int is[DIMS()];
		int os[DIMS()];


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
		implementation(scalar* array_, PARENT parent_, integers... ints) : parent(parent_), array(array_) {
			init<0>(ints...);
		}

		__BCinline__ const auto size()		 const  { return this->os[DIMS() - 1]; }
		__BCinline__ const auto innerShape() const 	{ return is; }
		__BCinline__ const auto outerShape() const 	{ return parent.outerShape(); }

		__BCinline__ const auto getIterator() const { return array; }
		__BCinline__	   auto getIterator()   	{ return array; }
	};
	};
}

#endif /* TENSOR_RESHAPE_H_ */
