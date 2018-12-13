/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_GEMM_H_
#define EXPRESSION_BINARY_GEMM_H_

#include "Expression_Base.h"
#include "Internal_BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"

namespace BC {
namespace et {

template<class lv, class rv, class system_tag_>
struct Binary_Expression<lv, rv, oper::gemm<system_tag_>>
: Expression_Base<Binary_Expression<lv, rv,  oper::gemm<system_tag_>>>, BLAS_FUNCTION {


    using scalar_t  = typename lv::scalar_t;
    using allocator_t = typename lv::allocator_t;
    using system_tag = system_tag_;
    using impl_l  = typename blas::implementation<system_tag>;
    using utility_l   = utility::implementation<system_tag>;

    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value,\
    		"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

    __BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
    __BCinline__ static constexpr int ITERATOR() { return 0; }

    lv left;
    rv right;

     Binary_Expression(lv left, rv right) : left(left), right(right) {}

    __BCinline__ const auto inner_shape() const {
    	return l_array<DIMS()>([&](int i) {
    		return i == 0 ? left.rows() : i == 1 ? right.cols() : 1;
    	});
    }
    __BCinline__ const auto block_shape() const {
    	return l_array<DIMS()>([&](int i) {
    		return i == 0 ? left.rows() : i == 1 ? size() : 1;
    	});
    }

    __BCinline__ int size() const { return left.rows() * right.cols(); }
    __BCinline__ int rows() const { return left.rows(); }
    __BCinline__ int cols() const { return right.cols(); }
    __BCinline__ int dimension(int i) const { return inner_shape()[i]; }
    __BCinline__ int block_dimension(int i) const { return block_shape()[i]; }

    __BCinline__ int M() const { return left.rows();  }
    __BCinline__ int N() const { return right.cols(); }
    __BCinline__ int K() const { return left.cols();  }

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
        auto alpha = utility_l::stack_allocate((scalar_t)alpha_mod);
        auto beta = utility_l::stack_allocate((scalar_t)beta_mod);

        //compute the scalar values if need be
        if (lv_scalar)
        	impl_l::scalar_mul(alpha, alpha, alpha_lv);
        if (rv_scalar)
        	impl_l::scalar_mul(alpha, alpha, alpha_rv);

        //call matrix_mul
        impl_l::gemm(transA, transB,  M(), N(), K(), alpha, A, A.leading_dimension(0), B, B.leading_dimension(0), beta, injection, injection.leading_dimension(0));


        //deallocate all the temporaries
        if (lv_eval) cc(A).deallocate();
        if (rv_eval) cc(B).deallocate();
        utility_l::deallocate(beta);
        utility_l::deallocate(alpha);
    }
};

}
}

//std::cout << "A is transposed" << transA << std::endl;
//if (transB)
//std::cout <<"B is transposed" << transB << std::endl;
//if (lv_scalar)
//std::cout << "A has scalar " <<lv_scalar << std::endl;
//if (rv_scalar)
//std::cout <<"B has scalar" << rv_scalar << std::endl;
//if (lv_eval)
//std::cout << "A instant eval" <<lv_eval << std::endl;
//if(rv_eval)
//std::cout <<"B instant eval " << rv_eval << std::endl;
//std::cout << "alpha modifier = " << alpha_mod << std::endl;
//std::cout << " beta_mod = " << beta_mod << std::endl;

//__BCinline__ auto _slice(int i) {
//    return Binary_Expression<lv, decltype(right._slice(i)), oper::gemv<allocator_t>>(left, right._slice(i));
//}
//__BCinline__ auto _slice_range(int from, int to) {
//    return Binary_Expression<lv, decltype(right._slice_range(from, to)), oper::gemm<allocator_t>>(left, right._slice_range(from, to));
//}

#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
