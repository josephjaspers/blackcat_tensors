/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_DOT_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_DOT_H_

#include "expression_template_base.h"
#include "tree_evaluator.h"
#include "blas_expression_template_traits.h"

namespace bc {
namespace tensors {
namespace exprs { 


template<class lv, class rv, class SystemTag>
struct Bin_Op<oper::dot<SystemTag>, lv, rv>:
		Expression_Base<Bin_Op<oper::dot<SystemTag>, lv, rv>>,
		Shape<0>,
		oper::dot<SystemTag> {

	static_assert(std::is_same<
			typename lv::value_type,
			typename rv::value_type>::value,
			"ValueType must be the same");

	static_assert(
			lv::tensor_dim == 1 &&
			(rv::tensor_dim == 1 || rv::tensor_dim ==0),
			"DOT DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

	using value_type = typename lv::value_type;
	using system_tag = SystemTag;

	static constexpr int tensor_dim  = 0;
	static constexpr int tensor_iterator_dim = 0;

	lv left;
	rv right;

	using Shape<0>::inner_shape;

	Bin_Op(lv left, rv right, oper::dot<system_tag> op=oper::dot<system_tag>()):
		left(left),
		right(right) {}

	static oper::dot<system_tag> get_operation() {
		return oper::dot<system_tag>();
	}

	template<class Core, int Alpha, int Beta, class Stream>
	void eval(Output_Data<Core, Alpha, Beta> output, Stream stream) const {
		static_assert(Core::tensor_dim == 0,"Output must be a scalar");

		using blas_tools = blas_expression_parser::Blas_Expression_Parser<system_tag>;

		auto X = greedy_evaluate(left, stream);
		auto Y = greedy_evaluate(right, stream);
		auto& out = output.data();

		//call outer product
		bc::blas::BLAS<system_tag>::dot(
				stream,
				X.rows(), out.data(),
				X.data(), X.leading_dim(0),
				Y.data(), Y.leading_dim(0));

		constexpr int beta_value = Beta == 0 ? 1 : Beta;
		constexpr bool lv_scalar = blas_expression_traits<lv>::is_scalar_multiplied::value;
		constexpr bool rv_scalar = blas_expression_traits<rv>::is_scalar_multiplied::value;

		if (lv_scalar || rv_scalar) {
			auto alpha_lv = blas_expression_traits<lv>::get_scalar(left);
			auto alpha_rv = blas_expression_traits<rv>::get_scalar(right);
			blas_tools::scalar_multiply(stream, out.data(), beta_value, alpha_lv, alpha_rv);
		} else if (beta_value != 1) {
			blas_tools::scalar_multiply(stream, out.data(), out.data(), beta_value);
		}
    }
};


} //ns BC
} //ns exprs
} //ns tensors



#endif /* FUNCTION_DOT_H_ */
