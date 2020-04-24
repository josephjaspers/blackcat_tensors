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
struct Bin_Op:
		Expression_Base<Bin_Op<Operation, Lv, Rv>>,
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
	Bin_Op(Lv lv, Rv rv, const Args&... args):
		Operation(args...),
		left(lv),
		right(rv) {}


	template<class=std::enable_if_t<
		!expression_traits<Lv>::requires_greedy_evaluation::value &&
		!expression_traits<Rv>::requires_greedy_evaluation::value>>
	BCINLINE
	auto operator [](int index) const {
		return Operation::operator()(left[index], right[index]);
	}


	template<class=std::enable_if_t<
		!expression_traits<Lv>::requires_greedy_evaluation::value &&
		!expression_traits<Rv>::requires_greedy_evaluation::value>>
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


//forward declare
template<class T>
struct blas_expression_traits;

namespace detail {

using bc::oper::Add;
using bc::oper::Sub;
using bc::oper::Add_Assign;
using bc::oper::Sub_Assign;
using bc::oper::Negation;
using bc::oper::Scalar_Mul;

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto mk_bin_op(Lv left, Rv right, Args&&... args) {
	return Bin_Op<Op,Lv, Rv>(left, right, std::forward<Args>(args)...);
}

template<class Op, class Lv, class Rv, class=void>
struct bin_expr_factory
{
	template<class... Args>
	static auto make(Lv lv, Rv rv, Args&&... args) {
		return Bin_Op<Op, Lv, Rv>(lv, rv, std::forward<Args>(args)...);
	}
};

/// y + (-x) -> y - x
template<class Lv, class Rv>
struct bin_expr_factory<Add, Lv, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(Lv lv, Un_Op<Negation, Rv> rv, Args&&... args) {
		return mk_bin_op<Sub>(lv, rv.array, std::forward<Args>(args)...);
	}
};

/// y - (-x) -> y + x
template<class Lv, class Rv>
struct bin_expr_factory<Sub, Lv, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(Lv lv, Un_Op<Negation, Rv> rv, Args&&... args) {
		return mk_bin_op<Add>(lv, rv.array, std::forward<Args>(args)...);
	}
};

/// (-y) + x -> x - y
template<class Lv, class Rv>
struct bin_expr_factory<Sub, Un_Op<Negation, Lv>, Rv>
{
	template<class... Args>
	static auto make(Un_Op<Negation, Lv> lv, Rv rv, Args&&... args) {
		return mk_bin_op<Sub>(rv, lv.array, std::forward<Args>(args)...);
	}
};

/// (-y) - (-x) -> x - y //specialized to handle ambiguous overload
template<class Lv, class Rv>
struct bin_expr_factory<Sub, Un_Op<Negation, Lv>, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(
			Un_Op<Negation, Lv> lv,
			Un_Op<Negation, Rv> rv,
			Args&&... args)
	{
		return mk_bin_op<Sub>(rv.array, lv.array, std::forward<Args>(args)...);
	}
};

/// (-y) + (-x) -> (-y) - x //specialized to handle ambiguous overload
template<class Lv, class Rv>
struct bin_expr_factory<Add, Un_Op<Negation, Lv>, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(
			Un_Op<Negation, Lv> lv, Un_Op<Negation, Rv> rv, Args&&... args)
	{
		return detail::mk_bin_op<Sub>(
				lv, rv.array, std::forward<Args>(args)...);
	}
};

/// y += (-x) -> y -= x
template<class Lv, class Rv>
struct bin_expr_factory<Add_Assign, Lv, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(Lv lv, Un_Op<Negation, Rv> rv, Args&&... args) {
		return mk_bin_op<Sub_Assign>(lv, rv.array, std::forward<Args>(args)...);
	}
};

/// y -= (-x) -> y += x
template<class Lv, class Rv>
struct bin_expr_factory<Sub_Assign, Lv, Un_Op<Negation, Rv>>
{
	template<class... Args>
	static auto make(Lv lv, Un_Op<Negation, Rv> rv, Args&&... args) {
		return mk_bin_op<Add_Assign>(lv, rv.array, std::forward<Args>(args)...);
	}
};

/// scalar * blas_op(a, b) -> blas_op(a * scalar, b) || blas_op(a, b * scalar)
template<class Lv, class Rv>
struct bin_expr_factory<
		bc::oper::Scalar_Mul, Lv, Rv, std::enable_if_t<
				expression_traits<Lv>::is_blas_expression::value !=
				expression_traits<Rv>::is_blas_expression::value>>
{
	static auto make(Lv lv, Rv rv, Scalar_Mul op=Scalar_Mul())
	{
		constexpr bool left_scalar = Lv::tensor_dim==0;
		auto scalar_expr = bc::traits::constexpr_ternary<left_scalar>(
			[=]() { return std::make_pair(lv, rv); },
			[=]() { return std::make_pair(rv, lv); }
		);

		auto scalar = scalar_expr.first;
		auto expr   = scalar_expr.second;

		using expr_lv_t = std::decay_t<decltype(expr.left)>;
		using expr_rv_t = std::decay_t<decltype(expr.right)>;
		using expr_op_t = std::decay_t<decltype(expr.get_operation())>;

		constexpr bool expr_left_is_scalar_multiplied =
				blas_expression_traits<expr_lv_t>::is_scalar_multiplied::value;

		constexpr bool expr_right_is_scalar_multiplied =
				blas_expression_traits<expr_rv_t>::is_scalar_multiplied::value;

		//TODO add support for when both left and right are scalar multiplied
		static_assert(
				!(expr_left_is_scalar_multiplied
						&& expr_right_is_scalar_multiplied),
			"Cannot apply scalar_multiplication to a blas_expression where"
			"both the left and right expressions of the blas expression"
			"are already scalar multiplied");

		return bc::traits::constexpr_ternary<!expr_left_is_scalar_multiplied>(
			[=]() {
				auto newexpr = mk_bin_op<Scalar_Mul>(scalar, expr.left);
				return mk_bin_op<expr_op_t>(newexpr, expr.right);
			},
			[=]() {
				auto newexpr = mk_bin_op<Scalar_Mul>(scalar, expr.right);
				return mk_bin_op<expr_op_t>(newexpr, expr.left);
			});
	}
};

}

template<class Op, class Lv, class Rv> BCHOT
auto make_bin_expr(Lv left, Rv right, Op oper) {
	return detail::bin_expr_factory<Op, Lv, Rv>::make(left, right, oper);
}

template<class Op, class Lv, class Rv, class... Args> BCHOT
auto make_bin_expr(Lv left, Rv right, Args&&... args)
{
	return detail::bin_expr_factory<Op, Lv, Rv>::make(
			left, right, std::forward<Args>(args)...);
}


} //ns BC
} //ns exprs
} //ns tensors

#include "blas_expression_template_traits.h"

#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */
