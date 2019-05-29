/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_

#include "Expression_Template_Base.h"
#include "Tree_Lazy_Evaluator.h"
#include "blas_tools/Blas_tools.h"


namespace BC {
namespace exprs {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::gemv<System_Tag>>
: Expression_Base<Binary_Expression<lv, rv,  oper::gemv<System_Tag>>>, oper::gemv<System_Tag> {

	static_assert(std::is_same<typename lv::value_type, typename rv::value_type>::value,
    		"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::tensor_dimension == 2 && rv::tensor_dimension == 1,
    		"GEMV DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    using value_type    = typename lv::value_type;
    using system_tag  	= System_Tag;
    using blas_impl     = BC::blas::implementation<system_tag>;
    using blas_util	    = BC::exprs::blas_tools::implementation<system_tag>;

    static constexpr bool transA = blas_expression_traits<lv>::is_transposed;
    static constexpr bool transB = blas_expression_traits<rv>::is_transposed;
    static constexpr bool lv_scalar = blas_expression_traits<lv>::is_scalar_multiplied;
    static constexpr bool rv_scalar = blas_expression_traits<rv>::is_scalar_multiplied;

    static constexpr int tensor_dimension = 1;
    static constexpr int tensor_iterator_dimension = 1;


    lv left;
    rv right;


     Binary_Expression(lv left, rv right)
    : left(left), right(right) {}

    BCINLINE BC::size_t size() const { return left.rows(); }
    BCINLINE BC::size_t rows() const { return left.rows(); }
    BCINLINE BC::size_t cols() const { return 1; }
    BCINLINE BC::size_t dimension(int i) const { return i == 0 ? rows() : 1; }
    BCINLINE BC::size_t block_dimension(int i) const { return i == 0 ? rows() : 1; }
    BCINLINE BC::size_t M() const { return left.rows(); }
    BCINLINE BC::size_t N() const { return left.cols(); }

    BCINLINE const auto inner_shape() const { return make_lambda_array<tensor_dimension>([&](int i) { return i == 0 ? left.rows() : 1; });}
    BCINLINE const auto block_shape() const { return make_lambda_array<tensor_dimension>([&](int i) { return i == 0 ? rows() : 1; });}

    template<class core, BC::size_t  alpha_mod, BC::size_t  beta_mod, class allocator>
    void eval(tree::injector<core, alpha_mod, beta_mod> injection_values, allocator& alloc) const {

		//get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
		auto& injection = injection_values.data();

		//evaluate the left and right branches (computes only if necessary)
        auto contents = blas_util::template parse_expression<alpha_mod, beta_mod>(alloc, left, right);
        auto A = contents.left;
        auto X = contents.right;
        auto alpha = contents.alpha;
        auto beta  = contents.beta;

		blas_impl::gemv(alloc, transA,  M(), N(),
				alpha, A, A.leading_dimension(0),
				X, X.leading_dimension(0)/*inc_X*/,
				beta,
				injection/*Y*/, injection.leading_dimension(0)/*incy*/);

        blas_util::post_parse_expression_evaluation(alloc, contents);

    }
};


}
}


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
