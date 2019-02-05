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
#include "BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"


namespace BC {
namespace et {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::dot<System_Tag>>
: Expression_Base<Binary_Expression<lv, rv,  oper::dot<System_Tag>>>, BLAS_Function, Shape<0> {

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value,
			"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::DIMS == 1 && (rv::DIMS == 1 || rv::DIMS ==0),
    		"DOT DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    using value_type  = typename lv::value_type;
    using allocator_t = typename lv::allocator_t;
    using system_tag = System_Tag;
    using impl_l  = typename blas::implementation<system_tag>;
    using utility_l = utility::implementation<system_tag>;
    using function_t = oper::dot<System_Tag>;

    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

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
		auto X = CacheEvaluator<allocator>::evaluate(blas_feature_detector<lv>::get_array(left), alloc);
		auto Y = CacheEvaluator<allocator>::evaluate(blas_feature_detector<rv>::get_array(right), alloc);

		//allocate the alpha and beta scalars,
        auto alpha = alloc.scalar_alpha((value_type)alpha_mod);

		//call outer product
		impl_l::dot(alloc, X.rows(), injection, X, X.leading_dimension(0), Y, Y.leading_dimension(0));

		if (lv_scalar) {
			auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
			impl_l::scalar_mul(injection.memptr(), alpha, alpha_lv, alloc);
		}
		if (rv_scalar) {
			auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);
			impl_l::scalar_mul(injection.memptr(), alpha, alpha_rv, alloc);
		}
		if (beta_mod) {
			auto beta = alloc.scalar_constant<value_type, beta_mod>();
			impl_l::scalar_mul(alpha, alpha, beta, alloc);
		}


		//deallocate all the temporaries
		if (lv_eval) cc(X).deallocate();
		if (rv_eval) cc(Y).deallocate();
	}
};


}
}



#endif /* FUNCTION_DOT_H_ */
