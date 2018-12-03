/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_BINARY_NONLINEAR_H_
#define PTEE_BINARY_NONLINEAR_H_

#include "Tree_Evaluator_Common.h"
#include "Expression_Binary.h"

namespace BC {
namespace et     {
namespace tree {

template<class lv, class rv, class op>
struct evaluator<Binary_Expression<lv, rv, op>, std::enable_if_t<is_nonlinear_op<op>()>> {
    static constexpr bool entirely_blas_expr = false;
    static constexpr bool partial_blas_expr = false;
    static constexpr bool nested_blas_expr = evaluator<lv>::nested_blas_expr || evaluator<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval = evaluator<lv>::requires_greedy_eval || evaluator<rv>::requires_greedy_eval;


    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        return branch;
    }


    struct left_trivial_injection {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::injection(branch.left, tensor);
            auto right = branch.right;
            return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
        }
    };
    struct right_trivial_injection {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = branch.left;
            auto right = evaluator<rv>::injection(branch.right, tensor);
            return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
        }
    };
    struct left_nontrivial_injection {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::injection(branch.left, tensor);
            auto right = branch.right; //rv
            return Binary_Expression<std::decay_t<decltype(left)>, rv, op>(left, right);
        }
    };
    struct right_nontrivial_injection {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = branch.left; //lv
            auto right = evaluator<rv>::injection(branch.right, tensor);
            return Binary_Expression<lv, std::decay_t<decltype(right)>, op>(left, right);
        }
    };

    template<class core, int a, int b> __BChot__
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            //dont need to update injection
            //trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
            using impl  =
                std::conditional_t<evaluator<lv>::partial_blas_expr,         left_trivial_injection,
                std::conditional_t<evaluator<rv>::partial_blas_expr,         right_trivial_injection,
                std::conditional_t<evaluator<lv>::nested_blas_expr,     left_nontrivial_injection,
                std::conditional_t<evaluator<rv>::nested_blas_expr,     right_nontrivial_injection, void>>>>;

            return impl::function(branch, tensor);

    }

    struct replacement_required {
        __BChot__
        static auto function(const Binary_Expression<lv,rv,op>& branch) {
            using branch_t = Binary_Expression<lv,rv,op>;
            auto tmp =  temporary<et::Array<branch_t::DIMS(), scalar_of<branch_t>, allocator_of<branch_t>>>(branch.inner_shape());
            auto inject_tmp = injector<std::decay_t<decltype(tmp)>, 1, 0>(tmp);
            return injection(branch, inject_tmp);
        }
    };
    struct replacement_not_required {
        __BChot__
        static auto function(const Binary_Expression<lv,rv,op>& branch) {
            return branch;
        }
    };

    __BChot__
    static auto temporary_injection(const Binary_Expression<lv,rv,op>& branch) {
    	auto left  = evaluator<lv>::temporary_injection(branch.left);
    	auto right = evaluator<rv>::temporary_injection(branch.right);
    	return make_bin_expr<op>(left, right);
    }
    __BChot__
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch) {
        evaluator<lv>::deallocate_temporaries(branch.left);
        evaluator<rv>::deallocate_temporaries(branch.right);
    }
};

}
}
}




#endif /* PTEE_BINARY_NONLINEAR_H_ */
