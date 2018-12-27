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

template<class Value, class System_Tag>
struct Unary_Expression<Value, oper::transpose<System_Tag>>
    : Expression_Base<Unary_Expression<Value, oper::transpose<System_Tag>>> {

    using value_type  = typename Value::value_type;
    using system_tag = System_Tag;
    using allocator_t = allocator::implementation<system_tag, value_type>;

    static constexpr int DIMS = Value::DIMS;
    static constexpr int ITERATOR = DIMS > 1? DIMS :0;

    Value array;

    Unary_Expression(Value p) : array(p) {}

    __BCinline__ const auto inner_shape() const {
        return l_array<DIMS>([=](int i) {
            if (DIMS >= 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i);
            else if (DIMS == 2)
                return i == 0 ? array.cols() : i == 1 ? array.rows() : 1;
            else if (DIMS == 1)
                return i == 0 ? array.rows() : 1;
            else
                return 1;
        });
    }
    __BCinline__ const auto block_shape() const {
        return l_array<DIMS>([=](int i) {
            return i == 0 ? array.cols() : 1 == 1 ? array.rows() : array.block_dimension(i);
        });
    }
    __BCinline__ auto operator [] (int i) const -> decltype(array[0]) {
        return array[i];
    }
    __BCinline__ BC::size_t  size() const { return array.size(); }
    __BCinline__ BC::size_t  rows() const { return array.cols(); }
    __BCinline__ BC::size_t  cols() const { return array.rows(); }

    __BCinline__ BC::size_t  dimension(int i) const { return i == 0 ? array.cols() : i == 1 ? array.rows() : array.dimension(i); }
    __BCinline__ BC::size_t  block_dimension(int i) const { return block_shape()[i]; }

    template<class... ints>
    __BCinline__ auto operator ()(int m, BC::size_t  n, ints... integers) const -> decltype(array(n,m)) {
        return array(n,m, integers...);
    }

};
}
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
