/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_BINARY_H_

#include <type_traits>
#include "expression_template_base.h"

namespace bc {
namespace tensors {
namespace exprs {

template<class Operation, class Lv, class Rv>
struct Binary_Expression:
		Expression_Base<Binary_Expression<Operation, Lv, Rv>>,
		Operation
{
	using system_tag = typename Lv::system_tag;
	using value_type = std::decay_t<decltype(
			std::declval<Operation>().operator()(
					std::declval<typename Lv::value_type>(),
					std::declval<typename Rv::value_type>()))>;

	static constexpr int tensor_dim =
			bc::traits::max(Lv::tensor_dim, Rv::tensor_dim);

private:

	static constexpr bool is_broadcast_expression =
			Lv::tensor_dim != Rv::tensor_dim &&
			Lv::tensor_dim != 0 &&
			Rv::tensor_dim != 0;

	static constexpr int max_dim = bc::traits::max(
					Lv::tensor_iterator_dim,
					Rv::tensor_iterator_dim,
					Lv::tensor_dim,
					Rv::tensor_dim);

	static constexpr int max_iterator = bc::traits::max(
					Lv::tensor_iterator_dim,
					Rv::tensor_iterator_dim);

	static constexpr bool continuous_mem_layout =
		Lv::tensor_iterator_dim <= 1 &&
		Rv::tensor_iterator_dim <= 1;

public:

	static constexpr int tensor_iterator_dim =
		is_broadcast_expression || !continuous_mem_layout ?
			max_dim :
			max_iterator;

	Lv left;
	Rv right;

	Operation get_operation() const {
		return static_cast<const Operation&>(*this);
	}

	template<class... Args> BCHOT
	Binary_Expression(Lv lv, Rv rv, const Args&... args):
		Operation(args...),
		left(lv),
		right(rv) {}


	BCINLINE auto operator [](int index) const {
		return Operation::operator()(left[index], right[index]);
	}

	BCINLINE
	auto operator [](int index) {
		return Operation::operator()(left[index], right[index]);
	}

	template<
		class... Integers,
		class=std::enable_if_t<
				(sizeof...(Integers)>=tensor_iterator_dim)>>
	BCINLINE
	auto  operator ()(Integers... ints) const {
		return Operation::operator()(left(ints...), right(ints...));
	}

	template<
		class... Integers,
		class=std::enable_if_t<(
				sizeof...(Integers)>=tensor_iterator_dim)>>
	BCINLINE
	auto operator ()(Integers... ints) {
	 	return Operation::operator()(left(ints...), right(ints...));
	}

private:

	BCINLINE
	const auto& shape() const {
		constexpr int max_dim = Lv::tensor_dim >= Rv::tensor_dim;
		return traits::get<max_dim>(right, left);
	}

public:

	BCINLINE bc::size_t size() const { return shape().size(); }
	BCINLINE bc::size_t rows() const { return shape().rows(); }
	BCINLINE bc::size_t cols() const { return shape().cols(); }
	BCINLINE bc::size_t dim(int i) const { return shape().dim(i); }
	BCINLINE auto inner_shape() const { return shape().inner_shape(); }
};


template<class Op, class Lv, class Rv> BCHOT
auto make_bin_expr(Lv left, Rv right, Op oper) {
	return Binary_Expression<Op,Lv, Rv>(left, right, oper);
}

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto make_bin_expr(Lv left, Rv right, Args&&... args) {
	return Binary_Expression<Op,Lv, Rv>(left, right, args...);
}


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */


