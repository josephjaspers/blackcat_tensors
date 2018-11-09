/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

#include "Tree_Evaluator_Array.h"
#include "Tree_Evaluator_Binary_Linear.h"
#include "Tree_Evaluator_Binary_NonLinear.h"
#include "Tree_Evaluator_Unary.h"
#include "Tree_Evaluator_BLAS.h"
#include "Tree_Evaluator_Temporary.h"

namespace BC{
namespace internal {
namespace tree {

struct Greedy_Evaluator {

//template<class lv, class rv, class op>
//static auto substitution_evaluate(Binary_Expression<lv, rv, op> expression);

struct sub_eval_recursion {
    template<class lv, class rv, class op> __BChot__
    static auto function(Binary_Expression<lv, rv, op> expression) {
        return substitution_evaluate(evaluator<Binary_Expression<lv, rv, op>>::replacement(expression));
    }
};
struct sub_eval_terminate {
    template<class lv, class rv, class op> __BChot__
    static auto function(Binary_Expression<lv, rv, op> expression) {
        return expression;
    }
};

template<class lv, class rv, class op> __BChot__
static auto substitution_evaluate(Binary_Expression<lv, rv, op> expression) {

#ifndef BC_NO_SUBSTITUTIONS
    using impl = std::conditional_t<evaluator<Binary_Expression<lv, rv, op>>::non_trivial_blas_injection,
                    sub_eval_recursion, sub_eval_terminate>;
    return impl::function(expression);
#else
    return expression;
#endif
}

template<class expression> __BChot__
static void deallocate_temporaries(expression expr) {
    evaluator<expression>::deallocate_temporaries(expr);
}

template<class lv, class rv> __BChot__
static  auto evaluate(Binary_Expression<lv, rv, oper::add_assign> expression) {
    constexpr bool fast_eval = tree::evaluator<rv>::trivial_blas_evaluation;
    auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, 1, 1>(expression.left));

    return MTF::constexpr_ternary<fast_eval>(
            [&]() { return expression.left;},
            [&]() {
                return substitution_evaluate(Binary_Expression<lv, std::decay_t<decltype(right)>, oper::add_assign>
                                    (expression.left, right));
            });
}
template<class lv, class rv> __BChot__
static auto evaluate(Binary_Expression<lv, rv, oper::sub_assign> expression) {

    constexpr bool fast_eval = tree::evaluator<rv>::trivial_blas_evaluation;
    auto right = evaluator<rv>::linear_evaluation(expression.right, injector<lv, -1, 1>(expression.left));

    return MTF::constexpr_ternary<fast_eval>(
            [&]() { return expression.left;},
            [&]() {
                return substitution_evaluate(Binary_Expression<lv, std::decay_t<decltype(right)>, oper::sub_assign>(expression.left, right));
            });
}
template<class lv, class rv> __BChot__
static auto evaluate(Binary_Expression<lv, rv, oper::assign> expression) {
    auto right = evaluator<rv>::injection(expression.right, injector<lv, 1, 0>(expression.left));

    constexpr bool fast_eval = std::is_same<std::decay_t<decltype(right)>, lv>::value;

    return MTF::constexpr_ternary<fast_eval>(
            [&]() { return expression.left;},
            [&]() {
                return substitution_evaluate(Binary_Expression<lv, std::decay_t<decltype(right)>, oper::assign>(expression.left, right));
            });
}
template<class lv, class rv> __BChot__
static auto evaluate(Binary_Expression<lv, rv, oper::mul_assign> expression) {
    return substitution_evaluate(expression);
}
template<class lv, class rv>__BChot__
static auto evaluate(Binary_Expression<lv, rv, oper::div_assign> expression) {
    return substitution_evaluate(expression);
}
};

}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
