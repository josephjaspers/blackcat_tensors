/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_GER_H_
#define EXPRESSION_BINARY_GER_H_

#include "Expression_Base.h"
#include "Internal_BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"


namespace BC {
namespace et {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::ger<System_Tag>>
    : Expression_Base<Binary_Expression<lv, rv,  oper::ger<System_Tag>>>, BLAS_FUNCTION {

    static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value,
    		"GER ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

    using value_type  = typename lv::value_type;
    using system_tag = System_Tag;
    using allocator_t = typename allocator::implementation<system_tag, value_type>;
    using blas_lib     = typename blas::implementation<system_tag>;
    using utility_lib  = typename utility::implementation<system_tag>;

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
    __BCinline__ BC::size_t  size() const { return left.size() * right.size(); }
    __BCinline__ BC::size_t  rows() const { return left.rows(); }
    __BCinline__ BC::size_t  cols() const { return right.cols(); }
    __BCinline__ BC::size_t  dimension(int i) const { return i == 0 ? rows() : i == 1 ? cols() : 1; }
    __BCinline__ BC::size_t  block_dimension(int i) const { return this->block_shape()(i); }

    __BCinline__ BC::size_t  outer_dimension() const { return rows(); }

    __BCinline__ const auto inner_shape() const { return l_array<DIMS>([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.rows() : 1; });}
    __BCinline__ const auto block_shape() const { return l_array<DIMS>([&](int i) { return i == 0 ? left.rows() : i == 1 ? size() : 1; });}
    __BCinline__ BC::size_t  M() const { return left.rows();  }
    __BCinline__ BC::size_t  N() const { return right.rows(); }


	template<class core, BC::size_t  alpha_mod, BC::size_t  beta_mod>
	void eval(tree::injector<core, alpha_mod, beta_mod> injection_values) const {

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//evaluate the left and right branches (computes only if necessary)
		auto A = CacheEvaluator<allocator_t>::evaluate(blas_feature_detector<lv>::get_array(left));
		auto B = CacheEvaluator<allocator_t>::evaluate(blas_feature_detector<rv>::get_array(right));

		//get the left and right side scalar values
		auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
		auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);

		//allocate the alpha and beta scalars,
		auto alpha = utility_lib::stack_allocate((value_type)alpha_mod);

		//compute the scalar values if need be
		if (lv_scalar)
			blas_lib::scalar_mul(alpha, alpha, alpha_lv);
		if (rv_scalar)
			blas_lib::scalar_mul(alpha, alpha, alpha_rv);

		//call outer product
		blas_lib::ger(M(), N(), alpha, A, A.leading_dimension(0), B, B.leading_dimension(0), injection, injection.leading_dimension(0));


		//deallocate all the temporaries
		if (lv_eval) cc(A).deallocate();
		if (rv_eval) cc(B).deallocate();
		utility_lib::deallocate(alpha);
	}
};


}
}


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
