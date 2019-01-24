/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_SLICE_H_
#define TENSOR_SLICE_H_

#include "Array_Base.h"

namespace BC {
namespace et     {

//Floored decrement just returns the max(param - 1, 0)

template<class PARENT>
struct Array_Slice
        : Array_Base<Array_Slice<PARENT>, PARENT::DIMS - 1>, Shape<PARENT::DIMS - 1> {

    using value_type = typename PARENT::value_type;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr BC::size_t  ITERATOR { return MTF::max(PARENT::ITERATOR - 1, 0); }
    __BCinline__ static constexpr BC::size_t  DIMS { return PARENT::DIMS - 1; }

    value_type* array_slice;

    __BCinline__ Array_Slice(PARENT parent_, BC::size_t  index)
    : Shape<PARENT::DIMS - 1> (parent_.get_shape()),
      array_slice(const_cast<value_type*>(parent_.slice_ptr(index))) {}

    __BCinline__ const value_type* memptr() const { return array_slice; }
    __BCinline__       value_type* memptr()       { return array_slice; }

    };

    template<class internal_t>
    auto make_slice(internal_t internal, BC::size_t  index) {
        return Array_Slice<internal_t>(internal, index);
    }
}
}
#endif /* TENSOR_SLICE_CU_ */
