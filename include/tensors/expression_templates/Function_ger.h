/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_

#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"
#include "Array_Scalar_Constant.h"
#include "blas_tools/Blas_tools.h"

namespace BC {
namespace tensors {
namespace exprs { 

template<class lv, class rv, class System_Tag>
struct Binary_Expression<oper::ger<System_Tag>, lv, rv>
    : Expression_Base<Binary_Expression<oper::ger<System_Tag>, lv, rv>>, oper::ger<System_Tag> {

	static_assert(std::is_same<typename lv::value_type, typename rv::value_type>::value,
    		"GER ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

    using value_type = typename lv::value_type;
    using system_tag = System_Tag;
    using blas_impl  = BC::blas::implementation<system_tag>;
    using blas_util	 = BC::tensors::exprs::blas_tools::implementation<system_tag>;

    static constexpr int tensor_dimension = 2;
    static constexpr int tensor_iterator_dimension = 1;

    static_assert(lv::tensor_dimension == 1 &&
    				rv::tensor_dimension == 1 &&
    				blas_expression_traits<rv>::is_transposed::value,
    				"GER DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    lv left;
    rv right;

    Binary_Expression(lv left, rv right) : left(left), right(right) {}
    BCINLINE BC::size_t  size() const { return left.size() * right.size(); }
    BCINLINE BC::size_t  rows() const { return left.rows(); }
    BCINLINE BC::size_t  cols() const { return right.cols(); }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? rows() : i == 1 ? cols() : 1; }
    BCINLINE BC::size_t  block_dimension(int i) const {  return i == 0 ? left.rows() : i == 1 ? size() : 1; }

    BCINLINE BC::size_t  M() const { return left.rows();  }
    BCINLINE BC::size_t  N() const { return right.cols(); }


	template<class core, int alpha_mod, int beta_mod, class Stream>
	void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {
    	static_assert(core::tensor_dimension==2, "Ger injection must be a matrix");

		auto& injection = injection_values.data();

        //if we need to negate or zero the output
		//If beta_mod != 1 consider using gemm (to enable zeroing/modifying the output)
		if (beta_mod != 1) {
			auto expr = make_bin_expr<oper::Assign>(injection, make_scalar_constant<value_type>(beta_mod));
			evaluate(expr, stream);
		}

		if (blas_expression_traits<lv>::is_scalar_multiplied::value ||
				blas_expression_traits<rv>::is_scalar_multiplied::value) {

	        auto contents = blas_util::template parse_expression<alpha_mod, beta_mod>(stream, left, right);
	        auto A = contents.left;
	        auto B = contents.right;
	        auto alpha = contents.alpha;
			blas_impl::ger(stream, left.rows(), right.cols(),
					alpha.memptr(), A.memptr(), A.leading_dimension(0),
					B.memptr(), B.leading_dimension(0),
					injection.memptr(), injection.leading_dimension(0));
	        blas_util::post_parse_expression_evaluation(stream, contents);
		} else {
			auto alpha = make_constexpr_scalar<BC::host_tag, (alpha_mod == 0 ? 1 : alpha_mod), value_type>();
			auto A = greedy_evaluate(blas_expression_traits<lv>::remove_blas_modifiers(left), stream);
			auto B = greedy_evaluate(blas_expression_traits<rv>::remove_blas_modifiers(right), stream);
			stream.set_blas_pointer_mode_host();
			blas_impl::ger(stream, left.rows(), right.cols(),
					alpha.memptr(), A.memptr(), A.leading_dimension(0),
					B.memptr(), B.leading_dimension(0),
					injection.memptr(), injection.leading_dimension(0));
		}
	}
};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
