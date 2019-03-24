/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_

#include "Function_dot.h"
#include "Expression_Base.h"
#include "BLAS_Feature_Detector.h"
#include "Tree_Lazy_Evaluator.h"


namespace BC {
namespace exprs {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::gemv<System_Tag>>
: Expression_Base<Binary_Expression<lv, rv,  oper::gemv<System_Tag>>>, oper::gemv<System_Tag> {

	static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value,
    		"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::DIMS == 2 && rv::DIMS == 1,
    		"GEMV DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    using value_type    = typename lv::value_type;
    using system_tag  = System_Tag;
    using impl_l      = blas::implementation<system_tag>;
    using utility_l   = utility::implementation<system_tag>;
    using function_t = oper::gemv<System_Tag>;

    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static constexpr int DIMS = 1;
    static constexpr int ITERATOR = 1;


    lv left;
    rv right;


     Binary_Expression(lv left, rv right)
    : left(left), right(right) {}

    BCINLINE BC::size_t  size() const { return left.rows(); }
    BCINLINE BC::size_t  rows() const { return left.rows(); }
    BCINLINE BC::size_t  cols() const { return 1; }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? rows() : 1; }
    BCINLINE BC::size_t  block_dimension(int i) const { return i == 0 ? rows() : 1; }

    BCINLINE const auto inner_shape() const { return make_lambda_array<DIMS>([&](int i) { return i == 0 ? left.rows() : 1; });}
    BCINLINE const auto block_shape() const { return make_lambda_array<DIMS>([&](int i) { return i == 0 ? rows() : 1; });}

    template<class core, BC::size_t  alpha_mod, BC::size_t  beta_mod, class allocator>
    void eval(tree::injector<core, alpha_mod, beta_mod> injection_values, allocator& alloc) const {

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//evaluate the left and right branches (computes only if necessary)
		auto A = CacheEvaluator<allocator>::evaluate(blas_feature_detector<lv>::get_array(left), alloc);
		auto X = CacheEvaluator<allocator>::evaluate(blas_feature_detector<rv>::get_array(right), alloc);

		//allocate the alpha and beta scalars,
        auto alpha = alloc.scalar_alpha((value_type)alpha_mod);
		auto beta = alloc.template scalar_constant<value_type, beta_mod>();

		//get the left and right side scalar values and
		//compute the scalar values if need be
		if (lv_scalar) {
			auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
			impl_l::calculate_alpha(alloc, alpha, alpha, alpha_lv);
		}
		if (rv_scalar) {
			auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);
			impl_l::calculate_alpha(alloc, alpha, alpha, alpha_rv);
		}

		//call matrix_mul ///for gemm we always use M, N, K regardless of transpose, but for gemv we always use pre-trans dimensions ???
		BC::size_t  m = A.rows();
		BC::size_t  n = A.cols();

		impl_l::gemv(alloc, transA,  m, n, alpha, A, A.leading_dimension(0), X, X.leading_dimension(0)/*inc_X*/, beta, injection/*Y*/, injection.leading_dimension(0)/*incy*/);

		//deallocate all the temporaries
		if (rv_eval) meta::bc_const_cast(X).deallocate();
		if (lv_eval) meta::bc_const_cast(A).deallocate();
    }
};


}
}


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
