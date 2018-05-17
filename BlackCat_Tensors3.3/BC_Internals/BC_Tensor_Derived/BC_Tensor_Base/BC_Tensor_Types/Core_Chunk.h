
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

template<class PARENT>
struct Tensor_Chunk  {

	template<int dimension>
	struct implementation : Core_Base<implementation<dimension>,dimension> {

	__BCinline__ static constexpr int DIMS() { return dimension; };
	__BCinline__ static constexpr int CONTINUOUS() { return DIMS(); }

	using scalar = _scalar<PARENT>;

	static_assert(PARENT::CONTINUOUS() == 0, "Tensor_Reshape may only reshape continuous tensors, you may attempt to copy instead");

//	static_assert(NEW_DIM > -1, "DIMENSIONALITY OF TENSOR MUST BE >= 0");

	PARENT parent;
	scalar* array;

	int is[DIMS()];
	int os[DIMS()];

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
	implementation(scalar* array_, PARENT parent_, integers... ints) : parent(parent_), array(array_) {
//		static_assert(sizeof...(integers) <= PARENT::DIMS(), "DIMENSIONALITY OF CHUNK BE LESS THAN OR EQUAL TO TENSOR_CORE_PARENT");
		init<0>(ints...);
//		if (this->size() >= parent.size()) {
//			std::cout << "TENSOR RESHAPE SIZE MUST BE EQUAL TO ITS ORIGINAL SIZE" << std::endl;
//			throw std::invalid_argument("INVALID CHUNK_SHAPE");
//		}
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
