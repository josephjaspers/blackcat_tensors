/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GER_H_

#include "Expression_Base.h"
#include "BLAS_Feature_Detector.h"
#include "Tree_Lazy_Evaluator.h"
#include "Array_Scalar_Constant.h"

namespace BC {
namespace exprs {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::ger<System_Tag>>
    : Expression_Base<Binary_Expression<lv, rv,  oper::ger<System_Tag>>>, oper::ger<System_Tag> {

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value,
    		"GER ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

    using value_type  = typename lv::value_type;
    using system_tag = System_Tag;
    using blas_lib     = typename blas::implementation<system_tag>;
    using utility_lib  = typename utility::implementation<system_tag>;
    using function_t = oper::ger<System_Tag>;

    static constexpr int DIMS = 2;
    static constexpr int ITERATOR = 1;

    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static_assert(lv::DIMS == 1 && rv::DIMS == 1 && transB,
    		"GER DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");


    lv left;
    rv right;


     Binary_Expression(lv left, rv right) : left(left), right(right) {}
    BCINLINE BC::size_t  size() const { return left.size() * right.size(); }
    BCINLINE BC::size_t  rows() const { return left.rows(); }
    BCINLINE BC::size_t  cols() const { return right.cols(); }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? rows() : i == 1 ? cols() : 1; }
    BCINLINE BC::size_t  block_dimension(int i) const { return this->block_shape()(i); }

    BCINLINE BC::size_t  outer_dimension() const { return rows(); }

    BCINLINE const auto inner_shape() const { return make_lambda_array<DIMS>([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.rows() : 1; });}
    BCINLINE const auto block_shape() const { return make_lambda_array<DIMS>([&](int i) { return i == 0 ? left.rows() : i == 1 ? size() : 1; });}
    BCINLINE BC::size_t  M() const { return left.rows();  }
    BCINLINE BC::size_t  N() const { return right.rows(); }


	template<class core, BC::size_t  alpha_mod, BC::size_t  beta_mod, class allocator>
	void eval(tree::injector<core, alpha_mod, beta_mod> injection_values, allocator& alloc) const {

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//evaluate the left and right branches (computes only if necessary)
		auto A = CacheEvaluator<allocator>::evaluate(blas_feature_detector<lv>::get_array(left), alloc);
		auto B = CacheEvaluator<allocator>::evaluate(blas_feature_detector<rv>::get_array(right), alloc);

		//allocate the alpha and beta scalars,
        auto alpha = alloc.scalar_alpha((value_type)alpha_mod);

        //if we need to negate or zero the output
		if (beta_mod != 1) {
			auto expr = make_bin_expr<oper::assign>(injection, scalar_constant<value_type>(beta_mod));
			evaluate(expr, alloc);
		}

		//compute the scalar values if need be
		if (lv_scalar) {
			auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
			blas_lib::scalar_mul(alloc, alpha, alpha, alpha_lv);
		}
		if (rv_scalar) {
			auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);
			blas_lib::scalar_mul(alloc, alpha, alpha, alpha_rv);
		}



		//call outer product
		blas_lib::ger(alloc, M(), N(), alpha, A, A.leading_dimension(0), B, B.leading_dimension(0), injection, injection.leading_dimension(0));


		//deallocate all the temporaries
		if (lv_eval) meta::bc_const_cast(A).deallocate();
		if (rv_eval) meta::bc_const_cast(B).deallocate();
	}
};


}
}


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
