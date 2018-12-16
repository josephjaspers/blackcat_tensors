/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ARRAY_SECTION_H_
#define ARRAY_SECTION_H_

#include "Array_Base.h"

namespace BC {
namespace et     {

template<class PARENT>
struct Array_Slice_Range
        : Array_Base<Array_Slice_Range<PARENT>,PARENT::DIMS()>,
          Shape<PARENT::DIMS()> {

    using scalar_t = typename PARENT::scalar_t;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr int ITERATOR() { return PARENT::ITERATOR(); }
    __BCinline__ static constexpr int DIMS()      { return PARENT::DIMS(); }

    scalar_t* array_slice;

    struct voidt;
    using vec_scalar_t = std::conditional_t<(DIMS() == 1), scalar_t, voidt>;
    using tensor_scalar_t = std::conditional_t<(DIMS() > 1), scalar_t, voidt>;

    __BCinline__ //specialization if not a vector
    Array_Slice_Range(PARENT parent_, int from, int to)
        : Shape<DIMS()>(parent_.as_shape()),
          array_slice(const_cast<scalar_t*>(parent_.slice_ptr(from))) {

        int range = to - from;
        int size = parent_.leading_dimension(DIMS() - 2) * range;
        this->m_inner_shape[DIMS() - 1] = range; //setting the outer_dimension
        this->m_outer_shape[DIMS() - 1] = size;  //adjusting the size
    }

    __BCinline__ //specialization if vector
    Array_Slice_Range(const vec_scalar_t* array, PARENT parent_, int range)
        : Shape<DIMS()>(parent_.inner_shape()), array_slice(const_cast<scalar_t*>(array)) {
        this->m_inner_shape[DIMS() - 1] = range;
    }

    __BCinline__ const scalar_t* memptr() const { return array_slice; }
    __BCinline__       scalar_t* memptr()       { return array_slice; }

};

	template<class internal_t>
	auto make_ranged_slice(internal_t parent, int from, int to) {
		return Array_Slice_Range<internal_t>(parent, from, to);
	}

}
}

#endif /* ARRAY_SECTION_H_ */
