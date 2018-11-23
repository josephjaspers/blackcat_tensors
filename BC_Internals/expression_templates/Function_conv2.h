/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef EXPRESSION_BINARY_CONV_N2_H_
#define EXPRESSION_BINARY_CONV_N2_H_

#include "Array_Base.h"
#include "BlackCat_Internal_Definitions.h"
#include "Expression_Base.h"
#include "BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Runner.h"

namespace BC {
namespace et     {
namespace oper {
template<int dimension, class ml> class conv : public BLAS_FUNCTION {};
}

/*
 * a = M x K
 * b = K x N
 * c = M x N
 */


template<class lv, class rv, class allocator>
struct Binary_Expression<lv, rv, oper::conv<2, allocator>>
: Expression_Base<Binary_Expression<lv, rv,  oper::conv<2, allocator>>>, BLAS_FUNCTION {

    using scalar_type = scalar_of<lv>;
    static constexpr bool transA = blas_feature_detector<lv>::transposed;
    static constexpr bool transB = blas_feature_detector<rv>::transposed;
    static constexpr bool lv_scalar = blas_feature_detector<lv>::scalar;
    static constexpr bool rv_scalar = blas_feature_detector<rv>::scalar;
    static constexpr bool lv_eval = blas_feature_detector<lv>::evaluate;
    static constexpr bool rv_eval = blas_feature_detector<rv>::evaluate;

    static_assert(std::is_same<scalar_of<lv>, scalar_of<rv>>::value, "MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");

    __BCinline__ static constexpr int DIMS() { return 2; }
    __BCinline__ static constexpr int ITERATOR() { return 0; }
    int size() const {
        return inner_shape()[0] * inner_shape()[1];
    }
    lv left;
    rv right;

     Binary_Expression(lv left, rv right) : left(left), right(right) {}

    __BCinline__ const auto inner_shape() const {
        return l_array<DIMS()>([&](int i) {
                    if (i == 0 )
                        return (left.rows() - right.rows() + 1);
                    else if (i == 1)
                        return (left.cols() - right.cols() + 1);
                    else
                        return 1;
                });
    }
    __BCinline__ const auto block_shape() const { return l_array<DIMS()>([&](int i) { return i == 0 ? left.rows() : i == 1 ? size() : 1; });}


    __BCinline__ int M() const { return left.rows();  }
    __BCinline__ int N() const { return right.cols(); }
    __BCinline__ int K() const { return left.cols();  }



template<class core, int alpha_mod, int beta_mod>
void eval(tree::injector<core, alpha_mod, beta_mod> injection_values) const {

    //get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
    auto& injection = injection_values.data();

    //evaluate the left and right branches (computes only if necessary)
    auto A = branched<allocator>::evaluate(left);
    auto B = branched<allocator>::evaluate(right);

    //call matrix_mul
    allocator::conv2(injection, A, B);

    //deallocate all the temporaries
    if (lv_eval) cc(A).deallocate();
    if (rv_eval) cc(B).deallocate();
}
};

}
}
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

#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
