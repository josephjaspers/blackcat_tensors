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

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class system_tag_>
struct Binary_Expression<lv, rv, oper::ger<system_tag_>>
    : Expression_Base<Binary_Expression<lv, rv,  oper::ger<system_tag_>>>, BLAS_FUNCTION {

    using scalar_t  = typename lv::scalar_t;
    using system_tag = system_tag_;
    using allocator_t = typename allocator::implementation<system_tag, scalar_t>;
    using blas_lib     = typename blas::implementation<system_tag>;
    using utility_lib  = typename utility::implementation<system_tag>;


    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
    static_assert(lv::DIMS() == 1 && rv::DIMS() == 1 && transB, "GER DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");
    __BCinline__ static constexpr int DIMS() { return 2; }
    __BCinline__ static constexpr int ITERATOR() { return 0; }

    lv left;
    rv right;

     Binary_Expression(lv left, rv right) : left(left), right(right) {}
    __BCinline__ int size() const { return left.size() * right.size(); }
    __BCinline__ int rows() const { return left.rows(); }
    __BCinline__ int cols() const { return right.cols(); }
    __BCinline__ int dimension(int i) const { return i == 0 ? rows() : i == 1 ? cols() : 1; }
    __BCinline__ int block_dimension(int i) const { return this->block_shape()(i); }

    __BCinline__ int outer_dimension() const { return rows(); }

    __BCinline__ const auto inner_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? right.rows() : 1; });}
    __BCinline__ const auto block_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? size() : 1; });}
    __BCinline__ int M() const { return left.rows();  }
    __BCinline__ int N() const { return right.rows(); }


template<class core, int alpha_mod, int beta_mod>
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
    auto alpha = utility_lib::stack_allocate((scalar_t)alpha_mod);

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

//        if (transA)
//        std::cout << "A is transposed" << transA << std::endl;
//        if (transB)
//        std::cout <<"B is transposed" << transB << std::endl;
//        if (lv_scalar)
//        std::cout << "A has scalar " <<lv_scalar << std::endl;
//        if (rv_scalar)
//        std::cout <<"B has scalar" << rv_scalar << std::endl;
//        if (lv_eval)
//        std::cout << "A instant eval" <<lv_eval << std::endl;
//        if(rv_eval)
//        std::cout <<"B instant eval " << rv_eval << std::endl;

//__BCinline__ auto _slice(int i) {
//    return Binary_Expression<lv, decltype(right._scalar(i)), oper::scalar_mul>(left, right._scalar(i));
//}
//__BCinline__ auto _slice_range(int from, int to) {
//    return Binary_Expression<lv, decltype(right._slice_range(from, to)), oper::ger<allocator_t>>(left, right._slice_range(from, to));
//}
//

