/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_

#include "expression_template_base.h"
#include "tree_evaluator.h"
#include "array_scalar_constant.h"
#include "blas_expression_template_traits.h"

namespace bc {
namespace tensors {
namespace exprs { 

template<class lv, class rv, class System_Tag>
struct Binary_Expression<oper::ger<System_Tag>, lv, rv>:
		Expression_Base<Binary_Expression<oper::ger<System_Tag>, lv, rv>>,
		oper::ger<System_Tag> {

	static_assert(
			std::is_same<
					typename lv::value_type,
					typename rv::value_type>::value,
			"GER arguments must have the same value_type");

	static_assert(lv::tensor_dim == 1 &&
			rv::tensor_dim == 1 &&
			blas_expression_traits<rv>::is_transposed::value,
			"GER DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

	using value_type = typename lv::value_type;
	using system_tag = System_Tag;

	static constexpr int tensor_dim = 2;
	static constexpr int tensor_iterator_dim = 1;

	lv left;
	rv right;

	Binary_Expression(lv left, rv right):
		left(left),
		right(right) {}

	static oper::ger<System_Tag> get_operation() {
		return oper::ger<System_Tag>();
	}

	BCINLINE bc::size_t  size() const { return left.size() * right.size(); }
	BCINLINE bc::size_t  dim(int i) const { return i == 0 ? left.rows() : i == 1 ? right.cols() : 1; }
	BCINLINE bc::size_t rows() const { return dim(0); }
	BCINLINE bc::size_t cols() const { return dim(1); }


	template<class core, int Alpha, int Beta, class Stream>
	void eval(Output_Data<core, Alpha, Beta> output, Stream stream) const {
		static_assert(core::tensor_dim==2, "Ger out must be a matrix");

		using self_t = Binary_Expression<oper::ger<System_Tag>, lv, rv>;
		using traits = blas_expression_traits<self_t>;

		auto& out = output.data();

		//if we need to negate or zero the output
		//If Beta != 1 consider using gemm (to enable zeroing/modifying the output)
		if (Beta != 1) {
			auto expr = make_bin_expr<oper::Assign>(out, make_scalar_constant<value_type>(Beta));
			evaluate(expr, stream);
		}

		if (blas_expression_traits<lv>::is_scalar_multiplied::value ||
				blas_expression_traits<rv>::is_scalar_multiplied::value) {

			auto contents = traits::template parse_expression<Alpha, Beta>(stream, *this);
			auto A = contents.left;
			auto B = contents.right;
			auto alpha = contents.alpha;
			bc::blas::BLAS<system_tag>::ger(stream, left.rows(), right.cols(),
					alpha.data(), A.data(), A.leading_dim(0),
					B.data(), B.leading_dim(0),
					out.data(), out.leading_dim(1));
			traits::post_parse_expression_evaluation(stream, contents);
		} else {
			auto alpha = make_constexpr_scalar<bc::host_tag, (Alpha == 0 ? 1 : Alpha), value_type>();
			auto A = greedy_evaluate(blas_expression_traits<lv>::remove_blas_modifiers(left), stream);
			auto B = greedy_evaluate(blas_expression_traits<rv>::remove_blas_modifiers(right), stream);
			stream.set_blas_pointer_mode_host();
			bc::blas::BLAS<system_tag>::ger(stream, left.rows(), right.cols(),
					alpha.data(), A.data(), A.leading_dim(0),
					B.data(), B.leading_dim(0),
					out.data(), out.leading_dim(1));
		}
	}
};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
