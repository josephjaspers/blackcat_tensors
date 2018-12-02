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


    //--------------------------------------------Linear evaluation branches----------------------------------------------//
    struct full_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	evaluator<lv>::linear_evaluation(branch.left, tensor);
        	evaluator<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
        	return tensor.data();
        }
    };
    struct left_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            evaluator<lv>::linear_evaluation(branch.left, tensor);
            return evaluator<rv>::linear_evaluation(branch.right, tensor);
        }
    };
    struct right_eval {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            evaluator<rv>::linear_evaluation(branch.right, update_injection<op, b != 0>(tensor));
            return evaluator<lv>::linear_evaluation(branch.left, tensor);
        }
    };
    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        using impl = std::conditional_t<entirely_blas_expr, full_eval,
            std::conditional_t<evaluator<lv>::entirely_blas_expr, left_eval, right_eval>>;

        return impl::function(branch, tensor);
    }

    //-----------------------------------partial blas expr branches -----------------------------------------//
    struct left_blas_expr {

        struct trivial_injection {
            template<class l, class r> __BChot__
            static auto function(const l& left, const r& right) {
                return left;
            }
        };

		struct non_trivial_injection {
			template<class LeftVal, class RightVal> __BChot__
			static auto function(const LeftVal& left, const RightVal& right) {
				return Binary_Expression<LeftVal, RightVal, op>(left, right);
			}
		};

        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::injection(branch.left, tensor);

            //check this ?
            auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));

            using impl = std::conditional_t<evaluator<rv>::entirely_blas_expr,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right);
        }
    };
    struct right_blas_expr {
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
            auto right = evaluator<rv>::injection(branch.right, tensor);

            using impl = std::conditional_t<evaluator<lv>::entirely_blas_expr,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right);
        }
    };

    //------------nontrivial injections, DO NOT UPDATE, scalars-mods will enact during elementwise-evaluator----------------//
    struct left_nested_blas_expr {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            auto left = evaluator<lv>::injection(branch.left, tensor);
            rv right = branch.right; //rv
            return Binary_Expression<std::decay_t<decltype(left)>, rv, op>(left, right);
        }
    };
    struct right_nested_blas_expr {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
            lv left = branch.left; //lv
            auto right = evaluator<rv>::injection(branch.right, tensor);
            return Binary_Expression<lv, std::decay_t<decltype(right)>, op>(left, right);
        }
    };
    template<class core, int a, int b> __BChot__
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {

        using impl =
                std::conditional_t<entirely_blas_expr, full_eval,
                std::conditional_t<evaluator<lv>::partial_blas_expr, left_blas_expr,
                std::conditional_t<evaluator<rv>::partial_blas_expr, right_blas_expr,
                std::conditional_t<evaluator<lv>::nested_blas_expr, left_nested_blas_expr,
                std::conditional_t<evaluator<rv>::nested_blas_expr, right_nested_blas_expr, void>>>>>;
        return impl::function(branch, tensor);
    }


    //----------------------------------------------------substitution implementation-------------------------------------------//
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
