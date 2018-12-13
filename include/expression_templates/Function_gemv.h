/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_GEMV_H_
#define EXPRESSION_BINARY_GEMV_H_

#include "Function_dot.h"
#include "Expression_Base.h"
#include "Internal_BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"

namespace BC {
namespace et {

template<class lv, class rv, class system_tag_>
struct Binary_Expression<lv, rv, oper::gemv<system_tag_>>
: Expression_Base<Binary_Expression<lv, rv,  oper::gemv<system_tag_>>>, BLAS_FUNCTION {

    using scalar_t    = typename lv::scalar_t;
    using system_tag  = system_tag_;
    using allocator_t = allocator::implementation<system_tag, scalar_t>;
    using impl_l      = blas::implementation<system_tag>;
    using utility_l   = utility::implementation<system_tag>;

    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::DIMS() == 2 && rv::DIMS() == 1, "GEMV DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
    __BCinline__ static constexpr int DIMS() { return 1; }
    __BCinline__ static constexpr int ITERATOR() { return 0; }

    lv left;
    rv right;

     Binary_Expression(lv left, rv right) : left(left), right(right) {}

    __BCinline__ int size() const { return left.rows(); }
    __BCinline__ int rows() const { return left.rows(); }
    __BCinline__ int cols() const { return 1; }
    __BCinline__ int dimension(int i) const { return i == 0 ? rows() : 1; }
    __BCinline__ int block_dimension(int i) const { return i == 0 ? rows() : 1; }

    __BCinline__ const auto inner_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : 1; });}
    __BCinline__ const auto block_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? rows() : 1; });}

template<class core, int alpha_mod, int beta_mod>
void eval(tree::injector<core, alpha_mod, beta_mod> injection_values) const {

    //get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
    auto& injection = injection_values.data();

    //evaluate the left and right branches (computes only if necessary)
    auto A = CacheEvaluator<allocator_t>::evaluate(blas_feature_detector<lv>::get_array(left));
    auto X = CacheEvaluator<allocator_t>::evaluate(blas_feature_detector<rv>::get_array(right));

    //allocate the alpha and beta scalars,
    auto alpha = utility_l::stack_allocate((scalar_t)alpha_mod);
    auto beta  = utility_l::stack_allocate((scalar_t)beta_mod);

    //get the left and right side scalar values and
    //compute the scalar values if need be
    if (lv_scalar) {
        auto alpha_lv = blas_feature_detector<lv>::get_scalar(left);
        impl_l::scalar_mul(alpha, alpha, alpha_lv);
    }
    if (rv_scalar) {
        auto alpha_rv = blas_feature_detector<rv>::get_scalar(right);
        impl_l::scalar_mul(alpha, alpha, alpha_rv);
    }

    //call matrix_mul ///for gemm we always use M, N, K regardless of transpose, but for gemv we always use pre-trans dimensions ???
    int m = A.rows();
    int n = A.cols();

    impl_l::gemv(transA,  m, n, alpha, A, A.leading_dimension(0), X, X.leading_dimension(0)/*inc_X*/, beta, injection/*Y*/, injection.leading_dimension(0)/*incy*/);

    //deallocate all the temporaries
    if (lv_eval) cc(A).deallocate();
    if (rv_eval) cc(X).deallocate();
    utility_l::deallocate(beta);
    utility_l::deallocate(alpha);
}
};

}
}

#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */

/*
 *     __BCinline__ auto _slice(int i) {
        return Binary_Expression<decltype(left._row(i)), decltype(right._slice(i)), oper::dot<allocator_t>>(left._row(i), right._slice(i));
    }
    __BCinline__ auto _slice_range(int from, int to) {
        return Binary_Expression<lv, decltype(right._slice_range(from, to)), oper::gemv<allocator_t>>(left, right._slice_range(from, to));
    }
 *
 */
