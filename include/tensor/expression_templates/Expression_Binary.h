/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_

#include "Expression_Template_Base.h"
#include "Shape.h"

namespace BC {
namespace tensors {
namespace exprs {

template<class Lv, class Rv, class Operation>
struct Binary_Expression : public Expression_Base<Binary_Expression<Lv, Rv, Operation>>, public Operation {

	using lv_value_t = typename Lv::value_type;
	using rv_value_t = typename Rv::value_type;
    using return_type = decltype(std::declval<Operation>()(std::declval<lv_value_t&>(), std::declval<rv_value_t&>()));
    using value_type  = std::remove_reference_t<std::decay_t<return_type>>;
    using system_tag  = typename Lv::system_tag;

    static constexpr int tensor_dimension = BC::meta::max(Lv::tensor_dimension, Rv::tensor_dimension);
    static constexpr int tensor_iterator_dimension = Lv::tensor_dimension != Rv::tensor_dimension ? tensor_dimension : BC::meta::max(Lv::tensor_iterator_dimension, Rv::tensor_iterator_dimension);

    Lv left;
    Rv right;

    Operation get_operation() const {
    	return static_cast<const Operation&>(*this);
    }

    template<class... args> BCHOT
    Binary_Expression(Lv l, Rv r, const args&... args_):
    Operation(args_...), left(l), right(r) {}


	BCINLINE auto  operator [](int index) const {
    	return Operation::operator()(left[index], right[index]);
    }

    template<class... integers>
    BCINLINE auto  operator ()(integers... ints) const {
    	return Operation::operator()(left(ints...), right(ints...));
    }

    BCINLINE
    auto  operator [](int index) {
    	return Operation::operator()(left[index], right[index]);
    }

    template<class... integers>
    BCINLINE auto  operator ()(integers... ints) {
     	return Operation::operator()(left(ints...), right(ints...));
    }

    auto dx() const {
		auto lv_dx = expression_traits<Lv>::select_on_dx(left);
		auto rv_dx = expression_traits<Rv>::select_on_dx(right);
		auto op_dx = oper::operation_traits<Operation>::select_on_dx(get_operation());
		using lv_dx_t = std::decay_t<decltype(lv_dx)>;
		using rv_dx_t = std::decay_t<decltype(rv_dx)>;
		using op_dx_t = std::decay_t<decltype(op_dx)>;
		static_assert(!std::is_same<BC::oper::Mul, Operation>::value, "Derivative of multiplication is not supported yet,"
																		"product rule is difficult to implement");
		return Binary_Expression<lv_dx_t, rv_dx_t, op_dx_t>(lv_dx, rv_dx, op_dx);
    }

private:
    BCINLINE const auto& shape() const { return dominant_type<Lv, Rv>::shape(left, right); }
public:
    BCINLINE BC::size_t size() const { return shape().size(); }
    BCINLINE BC::size_t rows() const { return shape().rows(); }
    BCINLINE BC::size_t cols() const { return shape().cols(); }
    BCINLINE BC::size_t dimension(int i) const { return shape().dimension(i); }
    BCINLINE BC::size_t block_dimension(int i) const { return shape().block_dimension(i); }
    BCINLINE const auto inner_shape() const { return shape().inner_shape(); }
    BCINLINE const auto block_shape() const { return shape().block_shape(); }
};

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto make_bin_expr(Lv left, Rv right, Args&&... args) {
	return Binary_Expression<
			std::decay_t<decltype(left.internal())>,
			std::decay_t<decltype(right.internal())>,
			Op>(left.internal(), right.internal(), args...);
}


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */


