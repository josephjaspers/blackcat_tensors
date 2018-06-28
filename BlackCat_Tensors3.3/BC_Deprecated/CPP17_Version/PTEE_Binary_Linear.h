/*
 * PTEE_Binary_Linear.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_BINARY_LINEAR_H_
#define PTEE_BINARY_LINEAR_H_

namespace BC {
namespace internal {
namespace tree {


template<class lv, class rv, class op>
struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
	static constexpr bool trivial_blas_eval = expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval;
	static constexpr bool trivial_blas_injection = expression_tree_evaluator<lv>::trivial_blas_injection || expression_tree_evaluator<rv>::trivial_blas_injection;
	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<lv>::non_trivial_blas_injection || expression_tree_evaluator<rv>::non_trivial_blas_injection;


	template<class core, int a, int b>
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval) {
			expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
			expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
			return tensor.data();
		}
		else if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval) {
			expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
			return expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
		}
		else { // (expression_tree_evaluator<rv>::trivial_blas_expr)
			expression_tree_evaluator<rv>::linear_evaluation(branch.left, update_injection<op>(tensor));
			return expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
		}
	}
	template<class core, int a, int b>
	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
			if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval) {
				expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
				expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
				return tensor.data();
			}
			//trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
			else if constexpr (expression_tree_evaluator<lv>::trivial_blas_injection) {
				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
				auto right = expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));

				//if trivial blas evaluation right hand side, remove the right side branch (return left)
				if constexpr(expression_tree_evaluator<rv>::trivial_blas_eval) {
					return left;
				} else {
					//else return the adjusted types
					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
				}

				//trivial injectable right hand side (prefer trivial injection right opposed to non-trivial)
			} else if constexpr (expression_tree_evaluator<rv>::trivial_blas_injection) {
				auto left = expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
				auto right = expression_tree_evaluator<rv>::injection(branch.right, update_injection<op>(tensor));

				//this branch will not get called, but this is to cover 'just-in-case' || showcase the parallels between lv/rv branches
				if constexpr(expression_tree_evaluator<lv>::trivial_blas_eval) {
					return right;
				} else {
					//this is what will be called
					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
				}
			}

			//no trivial injections on either side, attempt to use nested injection //attempt to do non trivial injection left-side
			//we do not need to use injection_update as the modifier will be evaluated when descending the tree (the branches are not removed)
			else if constexpr (expression_tree_evaluator<lv>::non_trivial_blas_injection) {

				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
				//this should be changed to substitution evaluation when possible
				auto right = branch.right; //rv
				return binary_expression<std::decay_t<decltype(left)>, rv, op>(left, right);
			}
			//no trivial injections on either side, attempt to use nested injection //attempt to do non trivial injection right-side
			else if constexpr (expression_tree_evaluator<rv>::non_trivial_blas_injection) {

				//this should be changed to substitution evaluation when possible
				auto left = branch.left; //lv
				auto right = expression_tree_evaluator<rv>::injection(branch.right, tensor);
				return binary_expression<lv, std::decay_t<decltype(right)>, op>(left, right);
			} else {
				throw std::invalid_argument("COMPLEX EVALUATOR FAILED TO DETECT AN INJECTION REPORT BUG");
			}
	}
	static auto replacement(const binary_expression<lv,rv,op>& branch) {
		if constexpr (non_trivial_blas_injection) {
			using branch_t = binary_expression<lv,rv,op>;
			auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
			return injection(branch, tmp);
		} else {
			return branch;
		}
	}
	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
		if constexpr (non_trivial_blas_injection) {
			expression_tree_evaluator<lv>::destroy_temporaries(branch.left);
			expression_tree_evaluator<rv>::destroy_temporaries(branch.right);
		} else
			return;
	}

};


}
}
}



#endif /* PTEE_BINARY_LINEAR_H_ */
