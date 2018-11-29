/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_BINARY_LINEAR_H_
#define PTEE_BINARY_LINEAR_H_

#include "Tree_Evaluator_Common.h"
#include "Expression_Binary.h"
#include "Array.h"

namespace BC {
namespace et     {
namespace tree {

template<class lv, class rv, class op>
struct evaluator<Binary_Expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
    static constexpr bool entirely_blas_expr = evaluator<lv>::entirely_blas_expr && evaluator<rv>::entirely_blas_expr;
    static constexpr bool partial_blas_expr = evaluator<lv>::partial_blas_expr || evaluator<rv>::partial_blas_expr;
    static constexpr bool nested_blas_expr = partial_blas_expr;

    struct full_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	branch.left.eval(tensor);
        	branch.right.eval(update_injection<op>(tensor));
            return tensor.data();
        }
    };
    struct left_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            evaluator<lv>::linear_evaluation(branch.left, tensor);
            return evaluator<rv>::linear_evaluation(branch.right, update_injection<op, partial_blas_expr>(tensor));
        }
    };
    struct right_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
            return evaluator<lv>::linear_evaluation(branch.left, tensor);
        }
    };

    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        static constexpr bool double_eval = evaluator<lv>::entirely_blas_expr && evaluator<rv>::entirely_blas_expr;

        using impl = std::conditional_t<double_eval, full_eval,
            std::conditional_t<evaluator<lv>::entirely_blas_expr, left_eval, right_eval>>;

        return impl::function(branch, tensor);
    }

    //------------------------------------------------------------------------
    struct left_trivial_injection {
        struct trivial_injection {
            template<class l, class r> __BChot__
            static auto function(const l& left, const r& right) {
                return left;
            }
        };
        struct non_trivial_injection {
                template<class l, class r> __BChot__
                static auto function(const l& left, const r& right) {
                    return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
                }
            };
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::injection(branch.left, tensor);
            auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));

            using impl = std::conditional_t<evaluator<rv>::entirely_blas_expr,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right);
        }
    };
    struct right_trivial_injection {
        struct trivial_injection {
            template<class l, class r> __BChot__
            static auto function(const l& left, const r& right) {
                return right;
            }
        };
        struct non_trivial_injection {
                template<class l, class r> __BChot__
                static auto function(const l& left, const r& right) {
                    return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
                }
            };
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::linear_evaluation(branch.left, tensor);
            auto right = evaluator<rv>::injection(branch.right, update_injection<op>(tensor));

            using impl = std::conditional_t<evaluator<lv>::entirely_blas_expr,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right);        }
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
        static constexpr bool full_eval_b= evaluator<lv>::entirely_blas_expr && evaluator<rv>::entirely_blas_expr;

        using impl =
                std::conditional_t<full_eval_b, full_eval,
                std::conditional_t<evaluator<lv>::partial_blas_expr, left_trivial_injection,
                std::conditional_t<evaluator<rv>::partial_blas_expr, right_trivial_injection,
                std::conditional_t<evaluator<lv>::nested_blas_expr, left_nontrivial_injection,
                std::conditional_t<evaluator<rv>::nested_blas_expr, right_nontrivial_injection, void>>>>>;
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
    static auto replacement(const Binary_Expression<lv,rv,op>& branch) {
        using impl = std::conditional_t<nested_blas_expr, replacement_required, replacement_not_required>;
        return impl::function(branch);
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



#endif /* PTEE_BINARY_LINEAR_H_ */
