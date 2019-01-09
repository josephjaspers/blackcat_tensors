/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_

#include "Expression_Base.h"

namespace BC {
namespace et {


template<class Value, class operation>
struct Unary_Expression : public Expression_Base<Unary_Expression<Value, operation>>, public operation {

    using value_type  = decltype(std::declval<operation>()(std::declval<typename Value::value_type>()));
    using system_tag  = typename Value::system_tag;
    using allocator_t = allocator::implementation<system_tag, value_type>;
    using utility_t	  = utility::implementation<system_tag>;
    using function_t  = operation;

    static constexpr int DIMS  = Value::DIMS;
    static constexpr int ITERATOR = Value::ITERATOR;

    Value array;

    template<class... args> __BCinline__
    Unary_Expression(Value v, const args&... args_)
    : operation(args_...) , array(v) {}

    __BCinline__
    auto operator [](int index) const {
        return static_cast<const operation&>(*this)(array[index]);
    }
    template<class... integers> __BCinline__
    auto operator ()(integers... index) const {
        return static_cast<const operation&>(*this)(array(index...));
    }

    __BCinline__  const auto inner_shape() const { return array.inner_shape(); }
    __BCinline__  const auto block_shape() const { return array.block_shape(); }
    __BCinline__ BC::size_t  size() const { return array.size(); }
    __BCinline__ BC::size_t  rows() const { return array.rows(); }
    __BCinline__ BC::size_t  cols() const { return array.cols(); }
    __BCinline__ BC::size_t  dimension(int i) const { return array.dimension(i); }
    __BCinline__ BC::size_t  block_dimension(int i) const { return array.block_dimension(i); }
};


template<class op, class expr> __BChot__
auto make_un_expr(expr e, op oper =op()) {
	return Unary_Expression<std::decay_t<decltype(e.internal())>, op>(e.internal(), oper);
}


}
}


#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
