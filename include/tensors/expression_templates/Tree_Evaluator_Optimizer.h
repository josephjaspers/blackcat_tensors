/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Tree_Struct_Injector.h"
#include "Array.h"
#include "Expression_Binary.h"
#include "Expression_Unary.h"

namespace BC {
namespace tensors {
namespace exprs { 

template<class T, class voider=void>
struct optimizer;

template<class T>
struct optimizer_default {
	/*
	 * entirely_blas_expr -- if we may replace this branch entirely with a temporary/cache
	 * partial_blas_expr  -- if part of this branch contains a replaceable branch nested inside it
	 * requires_greedy_eval   -- if a replaceable branch is inside a function (+=/-= won't work but basic assign = can work)
	 */

	static constexpr bool entirely_blas_expr = false;			//An expression of all +/- operands and BLAS calls				IE w*x + y*z
	static constexpr bool partial_blas_expr = false;			//An expression of element-wise +/- operations and BLAS calls	IE w + x*y
	static constexpr bool requires_greedy_eval = false;			//Basic check if any BLAS call exists at all

	template<class core, int a, int b, class StreamType>
	static auto linear_evaluation(T branch, Output_Data<core, a, b> tensor, StreamType) {
		return branch;
	}

	template<class core, int a, int b, class StreamType>
	static auto injection(T branch, Output_Data<core, a, b> tensor, StreamType) {
		return branch;
	}

	template<class StreamType>
	static auto temporary_injection(T branch, StreamType) {
		return branch;
	}

	template<class StreamType>
	static void deallocate_temporaries(T, StreamType) {
		return;
	}
};

template<class Op, class Array>
struct unary_optimizer_default {

	template<class StreamType>
	static auto temporary_injection(Unary_Expression<Op, Array> branch, StreamType stream) {
		auto expr = optimizer<Array>::temporary_injection(branch.array, stream);
		return make_un_expr(expr, branch.get_operation());

	}
	template<class StreamType>
	 static void deallocate_temporaries(Unary_Expression<Op, Array> branch, StreamType stream) {
		optimizer<Array>::deallocate_temporaries(branch.array, stream);
	}
};

template<class op, class lv, class rv>
struct binary_optimizer_default {

	template<class StreamType>
	static auto temporary_injection(Binary_Expression<op, lv, rv> branch, StreamType stream) {
		auto left  = optimizer<lv>::temporary_injection(branch.left, stream);
		auto right = optimizer<rv>::temporary_injection(branch.right, stream);
		return make_bin_expr<op>(left, right, branch.get_operation());
	}

	template<class StreamType>
	static void deallocate_temporaries(Binary_Expression<op, lv, rv> branch, StreamType stream) {
		optimizer<rv>::deallocate_temporaries(branch.right, stream);
		optimizer<lv>::deallocate_temporaries(branch.left, stream);
	}
};

//-------------------------------- Array ----------------------------------------------------//
template<class T>
struct optimizer<T, std::enable_if_t<
	expression_traits<T>::is_array::value &&
	!expression_traits<T>::is_temporary::value>>
: optimizer_default<T> {};

//--------------Temporary---------------------------------------------------------------------//

template<class Array>
struct optimizer<Array, std::enable_if_t<expression_traits<Array>::is_temporary::value>>
 : optimizer_default<Array> {

	template<class StreamType>
	static void deallocate_temporaries(Array tmp, StreamType stream) {
		using value_type = typename Array::value_type;
		tmp.deallocate(stream.template get_allocator_rebound<value_type>());
	}
};

//-----------------------------------------------BLAS----------------------------------------//

template<class Expression>
struct optimizer_greedy_evaluations {
	static constexpr bool entirely_blas_expr = true;
	static constexpr bool partial_blas_expr = true;
	static constexpr bool requires_greedy_eval = true;

private:

	template<class core, int a, int b, class StreamType>
	static auto evaluate_impl(
			Expression branch,
			Output_Data<core, a, b> tensor,
			StreamType stream,
			std::true_type valid_injection) {
		branch.eval(tensor, stream);
		return tensor.data();
	}

	template<class core, int a, int b, class StreamType>
	static auto evaluate_impl(
			Expression branch,
			Output_Data<core, a, b> tensor,
			StreamType stream,
			std::false_type valid_injection) {
		return branch;
	}

public:

	template<class core, int a, int b, class StreamType>
	static auto linear_evaluation(Expression branch, Output_Data<core, a, b> tensor, StreamType stream) {
		return evaluate_impl(branch, tensor, stream,
				BC::traits::truth_type<Expression::tensor_dimension == core::tensor_dimension>());
	}

	template<class core, int a, int b, class StreamType>
	static auto injection(Expression branch, Output_Data<core, a, b> tensor, StreamType stream) {
		return evaluate_impl(branch, tensor, stream,
				BC::traits::truth_type<Expression::tensor_dimension == core::tensor_dimension>());
	}

	template<class StreamType>
	static auto temporary_injection(Expression branch, StreamType stream) {
		using value_type = typename Expression::value_type;

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
	Binary_Expression<op, lv, rv>,
	std::enable_if_t<expression_traits<Binary_Expression<op, lv, rv>>::requires_greedy_evaluation::value>>:
	binary_optimizer_default<op, lv, rv>,
	optimizer_greedy_evaluations<Binary_Expression<op, lv, rv>> {

	using optimizer_greedy_evaluations<Binary_Expression<op, lv, rv>>::temporary_injection;
};

template<class op, class value>
struct optimizer<
	Unary_Expression<op, value>,
	std::enable_if_t<expression_traits<Unary_Expression<op, value>>::requires_greedy_evaluation::value>>:
	unary_optimizer_default<op, value>,
	optimizer_greedy_evaluations<Unary_Expression<op, value>> {

	using optimizer_greedy_evaluations<Unary_Expression<op, value>>::temporary_injection;
};


//-------------------------------- Linear ----------------------------------------------------//

template<class op, class lv, class rv>
struct optimizer<Binary_Expression<op, lv, rv>, std::enable_if_t<oper::operation_traits<op>::is_linear_operation>>:
 binary_optimizer_default<op, lv, rv> {

	static constexpr bool entirely_blas_expr 	=
			optimizer<lv>::entirely_blas_expr &&
			optimizer<rv>::entirely_blas_expr &&
			lv::tensor_dimension == rv::tensor_dimension;

	static constexpr bool partial_blas_expr =
			optimizer<lv>::partial_blas_expr ||
			optimizer<rv>::partial_blas_expr;

	static constexpr bool requires_greedy_eval =
			optimizer<lv>::requires_greedy_eval ||
			optimizer<rv>::requires_greedy_eval;

	template<class core, int a, int b, class StreamType>
	static auto linear_evaluation(Binary_Expression<op, lv, rv>& branch, Output_Data<core, a, b> tensor, StreamType stream) {
		return
				BC::traits::constexpr_if<entirely_blas_expr>(
						[&](){
							optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
							optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, true>(tensor), stream);
							return tensor.data();
						},
				BC::traits::constexpr_else_if<optimizer<lv>::entirely_blas_expr>(
						[&]() {
							return BC::traits::constexpr_ternary<std::is_same<op, oper::Sub>::value>(
									[&]() {
										/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
										auto right = optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, true>(tensor), stream);
										return make_un_expr<oper::negation>(right);
									},
									[&]() {
										/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
										auto right = optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, true>(tensor), stream);
										return right;
									}
							);
						},
				BC::traits::constexpr_else_if<optimizer<rv>::entirely_blas_expr>(
						[&]() {
								auto left  = optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
							  /*auto right = */ optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, b != 0>(tensor), stream);
								return left;
						},
				BC::traits::constexpr_else(
						[&]() {
							static constexpr bool left_evaluated = optimizer<lv>::partial_blas_expr || b != 0;
							auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
							auto right = optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, left_evaluated>(tensor), stream);
							return make_bin_expr<op>(left, right);
						}
				))));
	}

	template<class core, int a, int b, class StreamType>
	static auto injection(Binary_Expression<op, lv, rv> branch, Output_Data<core, a, b> tensor, StreamType stream) {

		auto basic_eval = [&]() {
			static constexpr bool left_evaluated = optimizer<lv>::partial_blas_expr || b != 0;
			auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
			auto right = optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, left_evaluated>(tensor), stream);
			return make_bin_expr<op>(left, right);
		};

		return
				BC::traits::constexpr_if<entirely_blas_expr>(
						[&](){
							optimizer<lv>::linear_evaluation(branch.left, tensor, stream);
							optimizer<rv>::linear_evaluation(branch.right, update_alpha_beta_modifiers<op, true>(tensor), stream);
							return tensor.data();
						},
				BC::traits::constexpr_else_if<optimizer<rv>::partial_blas_expr && optimizer<lv>::partial_blas_expr>(
							basic_eval,
				BC::traits::constexpr_else_if<optimizer<lv>::requires_greedy_eval>(
						[&]() {
							auto left = optimizer<lv>::injection(branch.left, tensor, stream);
							return make_bin_expr<op>(left, branch.right);
						},
				BC::traits::constexpr_else_if<optimizer<rv>::requires_greedy_eval>(
						[&]() {
							auto right = optimizer<rv>::injection(branch.right, tensor, stream);
							return make_bin_expr<op>(branch.left, right);
						},
				BC::traits::constexpr_else(
						basic_eval
				)))));
	}
};

//------------------------------Non linear-------------------------------------------//

template<class op, class lv, class rv>
struct optimizer<
Binary_Expression<op, lv, rv>, std::enable_if_t<
	oper::operation_traits<op>::is_nonlinear_operation &&
	!expression_traits<Binary_Expression<op, lv, rv>>::requires_greedy_evaluation::value>>:
 binary_optimizer_default<op, lv, rv> {

	static constexpr bool entirely_blas_expr = false;
	static constexpr bool partial_blas_expr = false;
	static constexpr bool requires_greedy_eval =
			optimizer<lv>::requires_greedy_eval ||
			optimizer<rv>::requires_greedy_eval;

	template<class core, int a, int b, class StreamType>
	static auto linear_evaluation(Binary_Expression<op, lv, rv> branch, Output_Data<core, a, b> tensor, StreamType) {
		return branch;
	}

	template<class core, int a, int b, class StreamType>
	static auto injection(Binary_Expression<op, lv, rv> branch, Output_Data<core, a, b> tensor, StreamType stream) {
		return BC::traits::constexpr_ternary<optimizer<lv>::partial_blas_expr || optimizer<lv>::requires_greedy_eval>(
					[&]() {
						auto left = optimizer<lv>::injection(branch.left, tensor, stream);
						auto right = branch.right;
						return make_bin_expr<op>(left, right, branch.get_operation());
					},
					[&]() {
						auto left = branch.left;
						auto right = optimizer<rv>::injection(branch.right, tensor, stream);
						return make_bin_expr<op>(left, right, branch.get_operation());
			}
		);
	}
};




//--------------Unary Expression---------------------------------------------------------------------//

template<class Op, class Array>
struct optimizer<Unary_Expression<Op, Array>,
std::enable_if_t<!expression_traits<Unary_Expression<Op, Array>>::requires_greedy_evaluation::value>>:
 unary_optimizer_default<Op, Array>
{
	static constexpr bool entirely_blas_expr 	= false;
	static constexpr bool partial_blas_expr 	= false;
	static constexpr bool requires_greedy_eval 	= optimizer<Array>::requires_greedy_eval;

	template<class core, int a, int b, class StreamType>
	static auto linear_evaluation(Unary_Expression<Op, Array> branch, Output_Data<core, a, b> tensor, StreamType) {
		return branch;
	}
	template<class core, int a, int b, class StreamType>
	static auto injection(Unary_Expression<Op, Array> branch, Output_Data<core, a, b> tensor, StreamType stream) {
		auto array =  optimizer<Array>::injection(branch.array, tensor, stream);
		return make_un_expr(array, branch.get_operation());
	}
};

} //ns exprs
} //ns tensors
} //ns BC



#endif /* PTE_ARRAY_H_ */
