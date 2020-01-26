/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "tree_output_data.h"
#include "array.h"
#include "expression_binary.h"
#include "expression_unary.h"

namespace bc {
namespace tensors {
namespace exprs { 

using bc::traits::truth_type;
using bc::traits::constexpr_if;
using bc::traits::constexpr_else_if;
using bc::traits::constexpr_else;
using bc::traits::constexpr_ternary;

template<class T, class voider=void>
struct optimizer;

template<class T>
struct optimizer_default
{
	/**
	 * entirely_blas_expr
	 *     if we may replace this branch entirely with a temporary/cache
	 *     expression is +/- ops and BLAS functions
	 *  Example -> w*x + y*z
	 *
	 * partial_blas_expr
	 *     if part of this branch contains a replaceable branch nested inside it
	 *     expression is +/- ops and BLAS functions OR simple cwisefunctions
	 *  Example ->  w + x*y
	 *
	 * requires_greedy_eval
	 *     does any non-lazy functions exist in the expression
	 * */

	static constexpr bool entirely_blas_expr = false;
	static constexpr bool partial_blas_expr = false;
	static constexpr bool requires_greedy_eval = false;

	template<class OutputData, class Stream>
	static auto linear_eval(T branch, OutputData, Stream) { return branch; }

	template<class OutputData, class Stream>
	static auto injection(T branch, OutputData, Stream) { return branch; }

	template<class Stream>
	static auto temporary_injection(T branch, Stream) { return branch; }

	template<class Stream>
	static void deallocate_temporaries(T, Stream) { return; }
};

template<class Op, class Array>
struct unary_optimizer_default
{
	template<class Stream>
	static auto temporary_injection(Un_Op<Op, Array> branch, Stream stream) {
		auto expr = optimizer<Array>::temporary_injection(branch.array, stream);
		return make_un_expr(expr, branch.get_operation());
	}

	template<class Stream>
	static void deallocate_temporaries(Un_Op<Op, Array> branch, Stream stream) {
		optimizer<Array>::deallocate_temporaries(branch.array, stream);
	}
};

template<class op, class lv, class rv>
struct binary_optimizer_default
{
	template<class Stream>
	static auto temporary_injection(Bin_Op<op, lv, rv> branch, Stream stream)
	{
		auto left  = optimizer<lv>::temporary_injection(branch.left, stream);
		auto right = optimizer<rv>::temporary_injection(branch.right, stream);
		return make_bin_expr<op>(left, right, branch.get_operation());
	}

	template<class Stream>
	static void deallocate_temporaries(Bin_Op<op, lv, rv> branch, Stream stream)
	{
		optimizer<rv>::deallocate_temporaries(branch.right, stream);
		optimizer<lv>::deallocate_temporaries(branch.left, stream);
	}
};

// -------------------------------- Array -------------------------------- //
template<class T>
struct optimizer<T, std::enable_if_t<
		expression_traits<T>::is_array::value &&
		!expression_traits<T>::is_temporary::value>>:
	optimizer_default<T> {};

// -------------------------------- Temp -------------------------------- //

template<class Array>
struct optimizer<
		Array,
		std::enable_if_t<expression_traits<Array>::is_temporary::value>>:
	optimizer_default<Array>
{
	template<class Stream>
	static void deallocate_temporaries(Array tmp, Stream stream)
	{
		using value_type = typename Array::value_type;
		tmp.deallocate(stream.template get_allocator_rebound<value_type>());
	}
};

// -------------------------------- Blas -------------------------------- //
template<class Xpr>
struct optimizer_greedy_evaluations
{
	static constexpr bool entirely_blas_expr = true;
	static constexpr bool partial_blas_expr = true;
	static constexpr bool requires_greedy_eval = true;

private:

	template<class OutputData, class Stream>
	static auto evaluate_impl(Xpr branch, OutputData tensor, Stream stream,
			std::true_type valid_injection)
	{
		branch.eval(tensor, stream);
		return tensor.data();
	}

	template<class OutputData, class Stream>
	static auto evaluate_impl(Xpr branch, OutputData tensor, Stream stream,
			std::false_type valid_injection)
	{
		return branch;
	}

public:

	template<class OutputData, class Stream>
	static auto linear_eval(Xpr branch, OutputData tensor, Stream stream) {
		return evaluate_impl(branch, tensor, stream,
				truth_type<Xpr::tensor_dim == OutputData::tensor_dim>());
	}

	template<class OutputData, class Stream>
	static auto injection(Xpr branch, OutputData tensor, Stream stream) {
		return evaluate_impl(branch, tensor, stream,
				truth_type<Xpr::tensor_dim == OutputData::tensor_dim>());
	}

	template<class Stream>
	static auto temporary_injection(Xpr branch, Stream stream)
	{
		using value_type = typename Xpr::value_type;

		auto temporary = make_kernel_array(
				branch.get_shape(),
				stream.template get_allocator_rebound<value_type>(),
				temporary_tag());

		branch.eval(make_output_data<1, 0>(temporary), stream);
		return temporary;
	}
};


template<class op, class lv, class rv>
struct optimizer<
		Bin_Op<op, lv, rv>,
		std::enable_if_t<
				expression_traits<Bin_Op<op, lv, rv>>
						::requires_greedy_evaluation::value>>:
	binary_optimizer_default<op, lv, rv>,
	optimizer_greedy_evaluations<Bin_Op<op, lv, rv>>
{
	using optimizer_greedy_evaluations<Bin_Op<op, lv, rv>>::temporary_injection;
};

template<class op, class value>
struct optimizer<
		Un_Op<op, value>,
		std::enable_if_t<
				expression_traits<Un_Op<op, value>>
						::requires_greedy_evaluation::value>>:
	unary_optimizer_default<op, value>,
	optimizer_greedy_evaluations<Un_Op<op, value>>
{
	using optimizer_greedy_evaluations<Un_Op<op, value>>::temporary_injection;
};


// ------------------------------ Linear ------------------------------//

template<class op, class lv, class rv>
struct optimizer<
		Bin_Op<op, lv, rv>,
		std::enable_if_t<oper::operation_traits<op>::is_linear_operation>>:
		binary_optimizer_default<op, lv, rv>
{
	static constexpr bool entirely_blas_expr =
			optimizer<lv>::entirely_blas_expr &&
			optimizer<rv>::entirely_blas_expr &&
			lv::tensor_dim == rv::tensor_dim;

	static constexpr bool partial_blas_expr =
			optimizer<lv>::partial_blas_expr ||
			optimizer<rv>::partial_blas_expr;

	static constexpr bool requires_greedy_eval =
			optimizer<lv>::requires_greedy_eval ||
			optimizer<rv>::requires_greedy_eval;

	template<class OutputData, class Stream>
	static
	auto linear_eval(
			Bin_Op<op, lv, rv>& branch, OutputData tensor, Stream stream)
	{
		auto rv_eval = [&](auto update_beta=std::true_type()) {
			using update_beta_t = std::decay_t<decltype(update_beta)>;
			return optimizer<rv>::linear_eval(
				branch.right,
				update_alpha_beta_modifiers<op, update_beta_t::value>(tensor),
				stream);
		};

		auto left = optimizer<lv>::linear_eval(branch.left, tensor, stream);

		return
		constexpr_if<entirely_blas_expr>(
			[&](){
				auto right = rv_eval(std::true_type());
				return tensor.data();
			},
		constexpr_else_if<optimizer<lv>::entirely_blas_expr>(
			[&]() {
				auto right = rv_eval(std::true_type());
				return constexpr_ternary<std::is_same<op, oper::Sub>::value>(
					[&]() {
						return make_un_expr<oper::Negation>(right);
					},
					[&]() {
						return right;
					}
				);
			},
		constexpr_else_if<optimizer<rv>::entirely_blas_expr>(
			[&]() {
					auto right = rv_eval(
							bc::traits::truth_type<OutputData::BETA>());
					return left;
			},
		constexpr_else(
			[&]() {
				using left_evaluated = truth_type<
						(optimizer<lv>::partial_blas_expr || OutputData::BETA)>;
				auto right = rv_eval(left_evaluated());
				return make_bin_expr<op>(left, right);
			}
		))));
	}

	template<class OutputData, class Stream> static
	auto injection(Bin_Op<op, lv, rv> branch, OutputData tensor, Stream stream)
	{
		auto lv_eval = [&]() {
			return optimizer<lv>::linear_eval(branch.left, tensor, stream);
		};

		auto rv_eval = [&](auto update_beta=std::true_type()) {
			using update_beta_t = std::decay_t<decltype(update_beta)>;
			return optimizer<rv>::linear_eval(
				branch.right,
				update_alpha_beta_modifiers<op, update_beta_t::value>(tensor),
				stream);
		};

		auto basic_eval = [&]()
		{
			using left_evaluated = truth_type<
					optimizer<lv>::partial_blas_expr || OutputData::BETA != 0>;
			return make_bin_expr<op>(lv_eval(), rv_eval(left_evaluated()));
		};

		return constexpr_if<entirely_blas_expr>(
			[&](){
				lv_eval();
				rv_eval(std::true_type());
				return tensor.data();
			},
		constexpr_else_if<
				optimizer<rv>::partial_blas_expr
				&& optimizer<lv>::partial_blas_expr>(
			basic_eval,
		constexpr_else_if<optimizer<lv>::requires_greedy_eval>(
			[&]() {
				auto left = optimizer<lv>::injection(branch.left, tensor, stream);
				return make_bin_expr<op>(left, branch.right);
			},
		constexpr_else_if<optimizer<rv>::requires_greedy_eval>(
			[&]() {
				auto right = optimizer<rv>::injection(branch.right, tensor, stream);
				return make_bin_expr<op>(branch.left, right);
			},
		constexpr_else(
				basic_eval
		)))));
	}
};

// ------------------------------ Non-linear ------------------------------//

template<class op, class lv, class rv>
struct optimizer<
		Bin_Op<op, lv, rv>,
		std::enable_if_t<
				oper::operation_traits<op>::is_nonlinear_operation &&
				!expression_traits<Bin_Op<op, lv, rv>>
						::requires_greedy_evaluation::value>>:
	binary_optimizer_default<op, lv, rv>
{
	static constexpr bool entirely_blas_expr = false;
	static constexpr bool partial_blas_expr = false;
	static constexpr bool requires_greedy_eval =
			optimizer<lv>::requires_greedy_eval ||
			optimizer<rv>::requires_greedy_eval;

	template<class OutputData, class Stream> static
	auto linear_eval(Bin_Op<op, lv, rv> branch, OutputData tensor, Stream) {
		return branch;
	}

	template<class OutputData, class Stream> static
	auto injection(Bin_Op<op, lv, rv> branch, OutputData tensor, Stream stream)
	{
		return constexpr_ternary<
			optimizer<lv>::partial_blas_expr ||
			optimizer<lv>::requires_greedy_eval>(
		[&]() {
			auto left = optimizer<lv>::injection(branch.left, tensor, stream);
			auto right = branch.right;
			return make_bin_expr<op>(left, right, branch.get_operation());
		}, [&]() {
			auto left = branch.left;
			auto right = optimizer<rv>::injection(branch.right, tensor, stream);
			return make_bin_expr<op>(left, right, branch.get_operation());
		});
	}
};

// ------------------------------ Un_Op ------------------------------//
template<class Op, class Array>
struct optimizer<
		Un_Op<Op, Array>,
		std::enable_if_t<!expression_traits<Un_Op<Op, Array>>
				::requires_greedy_evaluation::value>>:
	unary_optimizer_default<Op, Array>
{
	static constexpr bool entirely_blas_expr   = false;
	static constexpr bool partial_blas_expr    = false;
	static constexpr bool requires_greedy_eval = optimizer<Array>::requires_greedy_eval;

	template<class OutputData, class Stream> static
	auto linear_eval(Un_Op<Op, Array> branch, OutputData tensor, Stream) {
		return branch;
	}

	template<class OutputData, class Stream> static
	auto injection(Un_Op<Op, Array> branch, OutputData tensor, Stream stream)
	{
		auto array =  optimizer<Array>::injection(branch.array, tensor, stream);
		return make_un_expr(array, branch.get_operation());
	}
};

} //ns exprs
} //ns tensors
} //ns BC


#endif
