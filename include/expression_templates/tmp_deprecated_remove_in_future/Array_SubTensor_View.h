/*
 * Array_SubTensor_View.h
 *
 *  Created on: Dec 24, 2018
 *      Author: joseph
 */

#ifndef ARRAY_SUBTENSOR_VIEW_H_
#define ARRAY_SUBTENSOR_VIEW_H_

#include "Internal_Common.h"
#include "Array_Base.h"

namespace BC {
namespace et {

template<class PARENT, int ndimensions>
struct Array_SubTensor_View
        : Array_Base<Array_SubTensor_View<PARENT, ndimensions>, ndimensions>,
          Shape<ndimensions> {

    using value_type = typename PARENT::value_type;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr BC::size_t  ITERATOR { return ndimensions; }
    __BCinline__ static constexpr BC::size_t  DIMS { return ndimensions; }

    SubShape<ndimensions> m_shape;
    value_type* array;

    __BCinline__
    Array_SubTensor_View(PARENT parent_, BC::array<ndimensions, BC::size_t>& new_shape, BC::size_t index)
    : m_shape(new_shape, parent_.get_shape()),
      array(const_cast<value_type*>(parent_.slice_ptr(index))) {}

    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }

    };

    template<class internal_t, int dims = internal_t::DIMS>
    auto make_subtensor_view(internal_t internal, BC::size_t  index) {
        return Array_SubTensor_View<internal_t, dims>(internal, index);
    }
}
}



#endif /* ARRAY_SUBTENSOR_VIEW_H_ */
