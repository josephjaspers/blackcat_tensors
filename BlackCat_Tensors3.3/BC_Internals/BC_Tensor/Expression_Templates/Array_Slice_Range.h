/*
 * Array_Section.h
 *
 *  Created on: Aug 30, 2018
 *      Author: joseph
 */

#ifndef ARRAY_SECTION_H_
#define ARRAY_SECTION_H_

#include "Array_Base.h"

namespace BC {
namespace internal {

template<class PARENT>
struct Array_Slice_Range
		: Tensor_Array_Base<Array_Slice_Range<PARENT>,PARENT::DIMS()>,
		  Shape<PARENT::DIMS()> {

	using scalar_t = typename PARENT::scalar_t;
	using mathlib_t = typename PARENT::mathlib_t;

	__BCinline__ static constexpr int ITERATOR() { return PARENT::ITERATOR(); }
	__BCinline__ static constexpr int DIMS() 	 { return PARENT::DIMS(); }

	__BCinline__ operator const PARENT() const { return parent; }

	const PARENT parent;
	scalar_t* array_slice;

	__BCinline__ Array_Slice_Range(const scalar_t* array, PARENT parent_, int range)
							: Shape<DIMS()>(parent_.inner_shape()), parent(parent_), array_slice(const_cast<scalar_t*>(array)) {

		int size = parent.leading_dimension(DIMS() - 2) * range;
		this->IS[DIMS() - 1] = range; //setting the outer_dimension
		this->OS[DIMS() - 1] = size;  //adjusting the size
	}

	__BCinline__ const scalar_t* memptr() const { return array_slice; }
	__BCinline__ 	   scalar_t* memptr() 		{ return array_slice; }

};

}
}

#endif /* ARRAY_SECTION_H_ */
