/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_

#include "Array_Base.h"
namespace BC {
namespace et     {

template<class PARENT>
struct Array_Format
        : Array_Base<Array_Format<PARENT>, PARENT::DIMS>, Shape<PARENT::DIMS> {

    using value_type = typename PARENT::value_type;
    using allocator_t = typename PARENT::allocator_t;
    using system_tag = typename PARENT::system_tag;

    __BCinline__ static constexpr BC::size_t  DIMS          { return PARENT::DIMS; }
    __BCinline__ static constexpr BC::size_t  ITERATOR     { return DIMS; }

    value_type* array_slice;

     __BCinline__
    Array_Format(PARENT parent_, BC::array<DIMS - 1, int> format)
    : Shape<PARENT::DIMS> (parent_.as_shape()), array_slice(const_cast<value_type*>(parent_.memptr())) {

        for (int i = 0; i < format.size(); ++i) {
            this->m_inner_shape[i] = parent_.dimension(format[i] - 1);
            this->m_outer_shape[i] = parent_.leading_dimension(format[i] - 1);
        }
    }

    __BCinline__ const value_type* memptr() const { return array_slice; }
    __BCinline__       value_type* memptr()       { return array_slice; }

};

template<class internal_t, BC::size_t  dims>
auto make_format(internal_t internal, BC::array<dims, int> format) {
    return Array_Format<internal_t>(internal, format);
}

template<class internal> struct BC_lvalue_type_overrider<Array_Format<internal>> {
    static constexpr bool boolean = true;
};

}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_FORMAT_H_ */
