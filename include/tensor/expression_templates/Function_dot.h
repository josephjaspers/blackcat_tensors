/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_DOT_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_DOT_H_

#include "Expression_Base.h"
#include "Tree_Lazy_Evaluator.h"
#include "blas_tools/Blas_tools.h"


namespace BC {
namespace exprs {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::dot<System_Tag>>
: Expression_Base<Binary_Expression<lv, rv,  oper::dot<System_Tag>>>, Shape<0>, oper::dot<System_Tag> {

	static_assert(std::is_same<typename lv::value_type, typename rv::value_type>::value,
			"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::DIMS == 1 && (rv::DIMS == 1 || rv::DIMS ==0),
    		"DOT DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    using value_type = typename lv::value_type;
    using system_tag = System_Tag;
    using blas_impl  = BC::blas::implementation<system_tag>;
    using blas_util	 = BC::exprs::blas_tools::implementation<system_tag>;

    static constexpr bool lv_scalar = blas_expression_traits<lv>::is_scalar_multiplied;
    static constexpr bool rv_scalar = blas_expression_traits<rv>::is_scalar_multiplied;

    static constexpr int DIMS  = 0;
    static constexpr int ITERATOR = 0;

    lv left;
    rv right;

    Binary_Expression(lv left, rv right) : left(left), right(right) {}

    template<class core, BC::size_t  alpha_mod, BC::size_t  beta_mod, class allocator>
    void eval(tree::injector<core, alpha_mod, beta_mod> injection_values, allocator& alloc) const {

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//evaluate the left and right branches (computes only if necessary)
		//Note: dot does not accept a scalar Alpha, therefor we don't extract the array from left/right
		//The CacheEvaluator will generate a temporary if need be
		auto X = greedy_evaluate(left, alloc);
		auto Y = greedy_evaluate(right, alloc);

		//call outer product
		blas_impl::dot(alloc, X.rows(), injection, X, X.leading_dimension(0), Y, Y.leading_dimension(0));

		static constexpr int beta_value = beta_mod == 0 ? 1 : beta_mod;
		if (lv_scalar || rv_scalar) {
			auto alpha_lv = blas_expression_traits<lv>::get_scalar(left);
			auto alpha_rv = blas_expression_traits<rv>::get_scalar(right);
			blas_util::scalar_multiply(alloc, injection.memptr(), beta_value, alpha_lv, alpha_rv);
		} else if (beta_value != 1) {
			blas_util::scalar_multiply(alloc, injection.memptr(), injection.memptr(), beta_value);
		}
    }
};


}
}



#endif /* FUNCTION_DOT_H_ */
