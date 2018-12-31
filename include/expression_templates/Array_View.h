/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_

#include "Array_Base.h"
#include "Array.h"

namespace BC {
namespace et {


template<int Dimension, class Scalar, class Allocator>
struct Array_View
        : Array_Base<Array_View<Dimension, Scalar, Allocator>, Dimension>,
          Shape<Dimension> {

    using value_type = Scalar;
    using allocator_t = Allocator;
    using system_tag = typename BC::allocator_traits<allocator_t>::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;

    const value_type* array = nullptr;

    Array_View()                   = default;
    Array_View(const Array_View& ) = default;
    Array_View(      Array_View&&) = default;

    void copy_init(const Array_View& view) {
        this->copy_shape(view);
        this->array = view.array;
    }

    void swap_init(Array_View& array_move) {
		std::swap(this->array, array_move.array);
		this->swap_shape(array_move);
	}

    template<class tensor_t, typename = std::enable_if_t<tensor_t::DIMS == Dimension>>
    Array_View(const Array_Base<tensor_t, Dimension>& tensor)
        :  array(tensor) {

        this->copy_shape(static_cast<const tensor_t&>(tensor));
    }

    template<class... integers>
    Array_View(int x, integers... ints) :Shape<Dimension>(x, ints...) {}
    __BCinline__ const value_type* memptr() const  { return array; }

    void deallocate() {}
};
}
}

#endif /* ARRAY_VIEW_H_ */
