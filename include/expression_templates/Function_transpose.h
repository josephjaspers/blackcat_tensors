/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#define EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#include <vector>
#include "Expression_Base.h"

namespace BC {
namespace et {

template<class functor_type, class system_tag_>
struct Unary_Expression<functor_type, oper::transpose<system_tag_>>
    : Expression_Base<Unary_Expression<functor_type, oper::transpose<system_tag_>>> {

    using scalar_t  = typename functor_type::scalar_t;
    using system_tag = system_tag_;
    using allocator_t = allocator::implementation<system_tag, scalar_t>;

    __BCinline__ static constexpr int DIMS() { return functor_type::DIMS(); }
    __BCinline__ static constexpr int ITERATOR() { return DIMS() > 1? DIMS() :0; }

    functor_type array;

    Unary_Expression(functor_type p) : array(p) {}

    __BCinline__ const auto inner_shape() const {
        return l_array<DIMS()>([=](int i) {
            if (DIMS() >= 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i);
            else if (DIMS() == 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : 1;
            else if (DIMS() == 1)
                return i == 0 ? array.rows() : 1;
            else
                return 1;
        });
    }
    __BCinline__ const auto block_shape() const {
        return l_array<DIMS()>([=](int i) {
            return i == 0 ? array.cols() : 1 == 1 ? array.rows() : array.block_dimension(i);
        });
    }
    __BCinline__ auto operator [] (int i) const -> decltype(array[0]) {
        return array[i];
    }
    __BCinline__ int size() const { return array.size(); }
    __BCinline__ int rows() const { return array.cols(); }
    __BCinline__ int cols() const { return array.rows(); }

    __BCinline__ int dimension(int i) const { return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i); }
    __BCinline__ int block_dimension(int i) const { return block_shape()[i]; }

    template<class... ints>
    __BCinline__ auto operator ()(int m, int n, ints... integers) const -> decltype(array(n,m)) {
        return array(n,m, integers...);
    }

};
}
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
