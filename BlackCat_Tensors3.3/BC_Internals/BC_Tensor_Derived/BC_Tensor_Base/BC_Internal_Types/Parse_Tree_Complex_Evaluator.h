/*
 * Parse_Tree_Complex_Evaluator.h
 *
 *  Created on: Jun 18, 2018
 *      Author: joseph
 */

#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
#define PARSE_TREE_COMPLEX_EVALUATOR_H_

namespace BC{

namespace oper {
	struct add;
	struct sub;
	struct add_assign;
	struct sub_assign;
}

namespace internal {
namespace tree {


//trivial_blas_eval -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
template<class op, class core, int a, int b>//only apply update if right hand side branch
auto update_injection(injection_wrapper<core,a,b> tensor) {
	static constexpr int alpha_modifier = a != 0 ? a * alpha_of<op>() : 1;
	static constexpr int beta_modifier = b != 0 ? b * beta_of<op>() : 1;
	return injection_wrapper<core, alpha_modifier, beta_modifier>(tensor.data());
}

//disable default implementation (specialize for each type to ensure correct compilation)
template<class T, class enabler = void>
struct expression_tree_evaluator;

template<class T>
struct expression_tree_evaluator<T, std::enable_if_t<isCore<T>()>>
{
	static constexpr bool trivial_blas_eval = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = false;

	template<class core, int a, int b> __BC_host_inline__
	static auto linear_evaluation(const T& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b> __BC_host_inline__
	static auto injection(const T& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
};

template<class array_t, class op>
struct expression_tree_evaluator<unary_expression<array_t, op>>
{
	static constexpr bool trivial_blas_eval = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<array_t>::non_trivial_blas_injection;

	template<class core, int a, int b> __BC_host_inline__
	static auto linear_evaluation(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b> __BC_host_inline__
	static auto injection(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
		auto array =  expression_tree_evaluator<array_t>::injection(branch.array, tensor);
		using array_t_evaluated = std::decay_t<decltype(array)>;

		return unary_expression<array_t_evaluated, op>(array);
	}

};

template<class lv, class rv, class op>
struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_nonlinear_op<op>()>> {
	static constexpr bool trivial_blas_eval = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<lv>::non_trivial_blas_injection || expression_tree_evaluator<rv>::non_trivial_blas_injection;


	template<class core, int a, int b> __BC_host_inline__
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b> __BC_host_inline__
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
};


template<class lv, class rv, class op>
struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
	static constexpr bool trivial_blas_eval = expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval;
	static constexpr bool trivial_blas_injection = expression_tree_evaluator<lv>::trivial_blas_injection || expression_tree_evaluator<rv>::trivial_blas_injection;
	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<lv>::non_trivial_blas_injection || expression_tree_evaluator<rv>::non_trivial_blas_injection;


	template<class core, int a, int b> __BC_host_inline__
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
	template<class core, int a, int b> __BC_host_inline__
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
};

template<class lv, class rv, class op>
struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
	static constexpr bool trivial_blas_eval = true;
	static constexpr bool trivial_blas_injection = true;
	static constexpr bool non_trivial_blas_injection = true;


	template<class core, int a, int b> __BC_host_inline__
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}
	template<class core, int a, int b> __BC_host_inline__
	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}
};

template<class lv, class rv> __BC_host_inline__
auto evaluate(binary_expression<lv, rv, oper::add_assign> expression) {
	auto right = expression_tree_evaluator<rv>::linear_evaluation(expression.right, injection_wrapper<lv, 1, 1>(expression.left));
	return binary_expression<lv, std::decay_t<decltype(right)>, oper::add_assign>(expression.left, right);
}
template<class lv, class rv> __BC_host_inline__
auto evaluate(binary_expression<lv, rv, oper::sub_assign> expression) {
	auto right = expression_tree_evaluator<rv>::linear_evaluation(expression.right, injection_wrapper<lv, -1, 1>(expression.left));
	return binary_expression<lv, std::decay_t<decltype(right)>, oper::sub_assign>(expression.left, right);
}
template<class lv, class rv>  __BC_host_inline__
auto evaluate(binary_expression<lv, rv, oper::assign> expression) {
	auto right = expression_tree_evaluator<rv>::injection(expression.right, injection_wrapper<lv, 1, 0>(expression.left));
	return binary_expression<lv, std::decay_t<decltype(right)>, oper::assign>(expression.left, right);
}



}
}
}



#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
