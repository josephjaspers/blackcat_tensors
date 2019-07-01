/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_UNARY_H_


#include "Expression_Template_Base.h"
#include "Shape.h"


namespace BC {
namespace tensors {
namespace exprs { 

template<class Value, class Operation>
struct Unary_Expression : public Expression_Base<Unary_Expression<Value, Operation>>, public Operation {

    using return_type  = decltype(std::declval<Operation>()(std::declval<typename Value::value_type>()));
    using value_type  = std::remove_reference_t<std::decay_t<return_type>>;

    using system_tag  = typename Value::system_tag;

    static constexpr int tensor_dimension  = Value::tensor_dimension;
    static constexpr int tensor_iterator_dimension = Value::tensor_iterator_dimension;

    Value array;

    Operation get_operation() const {
    	return static_cast<const Operation&>(*this);
    }

    auto dx() const {
    	auto array_dx = exprs::expression_traits<Value>::select_on_dx(array);
    	auto dx_op    = oper::operation_traits<Operation>::select_on_dx(get_operation());
    	return Unary_Expression<decltype(array_dx),
    							decltype(dx_op)>(array_dx, dx_op);
    }

    template<class... args> BCINLINE
    Unary_Expression(Value v, const args&... args_)
    : Operation(args_...) , array(v) {}

    BCINLINE auto operator [](int index) const {
        return Operation::operator()(array[index]);
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) const {
        return Operation::operator()(array(index...));
    }

    BCINLINE auto operator [](int index) {
        return Operation::operator()(array[index]);
    }
    template<class... integers> BCINLINE
    auto operator ()(integers... index) {
        return Operation::operator()(array(index...));
    }

    BCINLINE const auto inner_shape() const { return array.inner_shape(); }
    BCINLINE const auto block_shape() const { return array.block_shape(); }
    BCINLINE BC::size_t size() const { return array.size(); }
    BCINLINE BC::size_t rows() const { return array.rows(); }
    BCINLINE BC::size_t cols() const { return array.cols(); }
    BCINLINE BC::size_t dimension(int i) const { return array.dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return array.block_dimension(i); }

};

template<class op, class expr> BCHOT
auto make_un_expr(expr e, op oper =op()) {
	return Unary_Expression<std::decay_t<decltype(e.internal())>, op>(e.internal(), oper);
}



} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
