/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef ARRAY_VIEW_H_
#define ARRAY_VIEW_H_

#include "Array_Base.h"
#include "Array.h"

namespace BC {
namespace et {


template<int dimension, class scalar, class allocator>
struct Array_View
        : Array_Base<Array_View<dimension, scalar, allocator>, dimension>,
          Shape<dimension> {

    using scalar_t = scalar;
    using allocator_t = allocator;
    using system_tag = typename allocator_t::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;

    const scalar_t* array = nullptr;

    Array_View()                   = default;
    Array_View(const Array_View& ) = default;
    Array_View(      Array_View&&) = default;

    void swap_array(Array_View& tensor) {
        std::swap(array, tensor.array);
    }

    void copy_init(const Array_View& view) {
        this->copy_shape(view);
        this->array = view.array;
    }

    template<class tensor_t, typename = std::enable_if_t<tensor_t::DIMS() == dimension>>
    Array_View(const Array_Base<tensor_t, dimension>& tensor)
        :  array(tensor) {

        this->copy_shape(static_cast<const tensor_t&>(tensor));
    }

    template<class... integers>
    Array_View(int x, integers... ints) :Shape<dimension>(x, ints...) {}
    __BCinline__ const scalar_t* memptr() const  { return array; }

    void deallocate() {}
};
}
}

#endif /* ARRAY_VIEW_H_ */
