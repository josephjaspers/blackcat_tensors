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
#include "Expression_Unary.h"
#include "Expression_Binary.h"
#include "Array.h"

namespace BC {
namespace et     {
namespace tree {

template<class lv, class rv, class op>
struct evaluator<Binary_Expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
    static constexpr bool entirely_blas_expr 	= evaluator<lv>::entirely_blas_expr && evaluator<rv>::entirely_blas_expr;
    static constexpr bool partial_blas_expr 	= evaluator<lv>::partial_blas_expr || evaluator<rv>::partial_blas_expr;
    static constexpr bool nested_blas_expr 		= evaluator<lv>::nested_blas_expr || evaluator<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= evaluator<lv>::requires_greedy_eval || evaluator<rv>::requires_greedy_eval;


    //--------------------------------------------Linear evaluation branches----------------------------------------------//


    //needed by injection and linear_evaluation
    struct remove_branch {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_branch (entire)");

        	evaluator<lv>::linear_evaluation(branch.left, tensor);
        	evaluator<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
        	return tensor.data();
        }
    };

    //if left is entirely blas_expr
    struct remove_left_branch {

        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_left_branch");

        	/*auto left = */ evaluator<lv>::linear_evaluation(branch.left, tensor);
            auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
            return right;
            //            return Binary_Expression<decltype(left), decltype(right), op>(left, right);
        }
    };

    //if right is entirely blas_expr (or if no blas expr)

    struct remove_right_branch {

    	template<class core, int a, int b> __BChot__
    	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_right_branch");

    		auto left  = evaluator<lv>::linear_evaluation(branch.left, tensor);
          /*auto right = */ evaluator<rv>::linear_evaluation(branch.right, update_injection<op, b != 0>(tensor));
            return left;
            //            return Binary_Expression<decltype(left), decltype(right), op>(left, right);
        }
    };

    struct basic_eval {

        template<class core, int a, int b> __BChot__
    	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- basic_eval");

        	static constexpr bool left_evaluated = evaluator<lv>::partial_blas_expr || b != 0;
        	auto left = evaluator<lv>::linear_evaluation(branch.left, tensor);
            auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op, left_evaluated>(tensor));
            return Binary_Expression<decltype(left), decltype(right), op>(left, right);
        }
    };


    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: linear_evaluation");

        using impl =
        		std::conditional_t<entirely_blas_expr, remove_branch,
        		std::conditional_t<evaluator<lv>::entirely_blas_expr, remove_left_branch,
        		std::conditional_t<evaluator<rv>::entirely_blas_expr, remove_right_branch,
        		basic_eval>>>;

        return impl::function(branch, tensor);
    }

    //-----------------------------------partial blas expr branches -----------------------------------------//
    struct evaluate_branch {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- evaluate_branch");
        	auto left = evaluator<lv>::linear_evaluation(branch.left, tensor);
        	auto right = evaluator<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
        	return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
        }
    };


    struct left_blas_expr {

        struct trivial_injection {
            template<class l, class r> __BChot__
            static auto function(const l& left, const r& right) {
	        	BC_TREE_OPTIMIZER_STDOUT("-- trivial_injection");
            	return left;
            }
        };

		struct non_trivial_injection {
			template<class LeftVal, class RightVal> __BChot__
			static auto function(const LeftVal& left, const RightVal& right) {
	        	BC_TREE_OPTIMIZER_STDOUT("-- non_trivial_injection");
				return Binary_Expression<LeftVal, RightVal, op>(left, right);
			}
		};

        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- left_blas_expr");

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
            	BC_TREE_OPTIMIZER_STDOUT("-- trivial_injection");
                return right;
            }
        };
        struct non_trivial_injection {
                template<class l, class r> __BChot__
                static auto function(const l& left, const r& right) {
                	BC_TREE_OPTIMIZER_STDOUT("-- non_trivial_injection");
                    return Binary_Expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
                }
            };
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- right_blas_expr");

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
        	BC_TREE_OPTIMIZER_STDOUT("- left_nested_blas_expr");

            auto left = evaluator<lv>::injection(branch.left, tensor);
            rv right = branch.right; //rv
            return Binary_Expression<std::decay_t<decltype(left)>, rv, op>(left, right);
        }
    };
    struct right_nested_blas_expr {
        template<class core, int a, int b> __BChot__
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
        	BC_TREE_OPTIMIZER_STDOUT("- right_nested_blas_expr");

            lv left = branch.left; //lv
            auto right = evaluator<rv>::injection(branch.right, tensor);
            return Binary_Expression<lv, std::decay_t<decltype(right)>, op>(left, right);
        }
    };


    template<class core, int a, int b> __BChot__
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Injection");

    	struct to_linear_eval {
    		static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
    	    	BC_TREE_OPTIMIZER_STDOUT("-flip to linear_eval");

    			return linear_evaluation(branch, tensor);
    		}
    	};

        using impl =
          		std::conditional_t<entirely_blas_expr, remove_branch,
          		//check this
//        		std::conditional_t<evaluator<lv>::entirely_blas_expr, remove_left_branch,
//        		std::conditional_t<evaluator<rv>::entirely_blas_expr, remove_right_branch,
        		std::conditional_t<evaluator<rv>::partial_blas_expr && evaluator<lv>::partial_blas_expr, basic_eval,
        		std::conditional_t<evaluator<lv>::nested_blas_expr, left_nested_blas_expr,
                std::conditional_t<evaluator<rv>::nested_blas_expr, right_nested_blas_expr, basic_eval>>>>;

        static_assert(!std::is_same<void, impl>::value, "EXPRESSION_REORDERING COMPILATION FAILURE, USE 'ALIAS' AS A WORKAROUND");
        return impl::function(branch, tensor);
    }


    //----------------------------------------------------substitution implementation-------------------------------------------//

    static auto temporary_injection(const Binary_Expression<lv,rv,op>& branch) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Temporary Injection");

    	auto left  = evaluator<lv>::temporary_injection(branch.left);
    	auto right = evaluator<rv>::temporary_injection(branch.right);
    	return make_bin_expr<op>(left, right);
    }

    __BChot__
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Deallocate Temporaries");

        evaluator<lv>::deallocate_temporaries(branch.left);
        evaluator<rv>::deallocate_temporaries(branch.right);
    }

};


}
}
}



#endif /* PTEE_BINARY_LINEAR_H_ */
