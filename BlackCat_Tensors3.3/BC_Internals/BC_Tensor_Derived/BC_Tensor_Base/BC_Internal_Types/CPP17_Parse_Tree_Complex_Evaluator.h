///*
// * Parse_Tree_Complex_Evaluator.h
// *
// *  Created on: Jun 18, 018
// *      Author: joseph
// */
//
//#ifndef PARSE_TREE_COMPLEX_EVALUATOR_H_
//#define PARSE_TREE_COMPLEX_EVALUATOR_H_
//
//#include "BC_Utility/Temporary.h"
//
//namespace BC{
//
//namespace oper {
//	struct add;
//	struct sub;
//	struct add_assign;
//	struct sub_assign;
//}
//
//namespace internal {
//namespace tree {
//
////trivial_blas_eval -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
//template<class op, class core, int a, int b>//only apply update if right hand side branch
//auto update_injection(injection_wrapper<core,a,b> tensor) {
//	static constexpr int alpha_modifier = a != 0 ? a * alpha_of<op>() : 1;
//	static constexpr int beta_modifier = b != 0 ? b * beta_of<op>() : 1;
//	return injection_wrapper<core, alpha_modifier, beta_modifier>(tensor.data());
//}
//
////disable default implementation (specialize for each type to ensure correct compilation)
//template<class T, class enabler = void>
//struct expression_tree_evaluator;
//
//template<class T>
//struct expression_tree_evaluator<T, std::enable_if_t<isArray<T>()>>
//{
//	static constexpr bool trivial_blas_eval = false;
//	static constexpr bool trivial_blas_injection = false;
//	static constexpr bool non_trivial_blas_injection = false;
//
//	template<class core, int a, int b>
//	static auto linear_evaluation(const T& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	template<class core, int a, int b>
//	static auto injection(const T& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	static auto replacement(const T& branch) {
//		return branch;
//	}
//	static void destroy_temporaries(const T& tmp) {
//		return;
//	}
//};
//
//template<class core>
//struct expression_tree_evaluator<temporary<core>>
//{
//	static constexpr bool trivial_blas_eval = false;
//	static constexpr bool trivial_blas_injection = false;
//	static constexpr bool non_trivial_blas_injection = false;
//
//	template<int a, int b>
//	static auto linear_evaluation(const temporary<core>& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	template<int a, int b>
//	static auto injection(const temporary<core>& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	static auto replacement(const temporary<core>& branch) {
//		return branch;
//	}
//	static void destroy_temporaries(temporary<core> tmp) {
//		tmp.destroy();
//	}
//};
//
//
//template<class array_t, class op>
//struct expression_tree_evaluator<unary_expression<array_t, op>>
//{
//	static constexpr bool trivial_blas_eval = false;
//	static constexpr bool trivial_blas_injection = false;
//	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<array_t>::non_trivial_blas_injection;
//
//	template<class core, int a, int b>
//	static auto linear_evaluation(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	template<class core, int a, int b>
//	static auto injection(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
//		auto array =  expression_tree_evaluator<array_t>::injection(branch.array, tensor);
//		using array_t_evaluated = std::decay_t<decltype(array)>;
//
//		return unary_expression<array_t_evaluated, op>(array);
//	}
//	//keep calling replacement till all the replacements are needed
//	static auto replacement(const unary_expression<array_t, op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			using branch_t = unary_expression<array_t, op>;
//			auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
//			return injection(branch, tmp);
//		} else {
//			return branch;
//		}
//	}
//	static void destroy_temporaries(const unary_expression<array_t, op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			expression_tree_evaluator<array_t>::destroy_temporaries(branch.array);
//		} else
//			return;
//	}
//};
//
//template<class lv, class rv, class op>
//struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_nonlinear_op<op>()>> {
//	static constexpr bool trivial_blas_eval = false;
//	static constexpr bool trivial_blas_injection = false;
//	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<lv>::non_trivial_blas_injection || expression_tree_evaluator<rv>::non_trivial_blas_injection;
//
//
//	template<class core, int a, int b>
//	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//		return branch;
//	}
//	template<class core, int a, int b>
//	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//			//dont need to update injection
//			//trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
//			if constexpr (expression_tree_evaluator<lv>::trivial_blas_injection) {
//
//				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
//				auto right = branch.right;
//
//				return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
//			}
//			//trivial injectable right hand side (prefer trivial injection right opposed to non-trivial)
//			else if constexpr (expression_tree_evaluator<rv>::trivial_blas_injection) {
//				auto left = branch.left;
//				auto right = expression_tree_evaluator<rv>::injection(branch.right, tensor);
//
//				return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
//			}
//
//			else if constexpr (expression_tree_evaluator<lv>::non_trivial_blas_injection) {
//
//				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
//				auto right = branch.right; //rv
//				return binary_expression<std::decay_t<decltype(left)>, rv, op>(left, right);
//			}
//			//no trivial injections on either side, attempt to use nested injection //attempt to do non trivial injection right-side
//			else if constexpr (expression_tree_evaluator<rv>::non_trivial_blas_injection) {
//
//				//this should be changed to substitution evaluation when possible
//				auto left = branch.left; //lv
//				auto right = expression_tree_evaluator<rv>::injection(branch.right, tensor);
//				return binary_expression<lv, std::decay_t<decltype(right)>, op>(left, right);
//			} else {
//				throw std::invalid_argument("COMPLEX EVALUATOR FAILED TO DETECT AN INJECTION REPORT BUG");
//			}
//	}
//	static auto replacement(const binary_expression<lv,rv,op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			using branch_t = binary_expression<lv,rv,op>;
//			auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
//			auto inject_tmp = injection_wrapper<std::decay_t<decltype(tmp)>, 1, 0>(tmp);
//			return injection(branch, inject_tmp);
//		} else {
//			return branch;
//		}
//	}
//	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			expression_tree_evaluator<lv>::destroy_temporaries(branch.left);
//			expression_tree_evaluator<rv>::destroy_temporaries(branch.right);
//		} else
//			return;
//	}
//};
//
//
//template<class lv, class rv, class op>
//struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_linear_op<op>()>> {
//	static constexpr bool trivial_blas_eval = expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval;
//	static constexpr bool trivial_blas_injection = expression_tree_evaluator<lv>::trivial_blas_injection || expression_tree_evaluator<rv>::trivial_blas_injection;
//	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<lv>::non_trivial_blas_injection || expression_tree_evaluator<rv>::non_trivial_blas_injection;
//
//
//	template<class core, int a, int b>
//	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//		if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval) {
//			expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
//			expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
//			return tensor.data();
//		}
//		else if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval) {
//			expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
//			return expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
//		}
//		else { // (expression_tree_evaluator<rv>::trivial_blas_expr)
//			expression_tree_evaluator<rv>::linear_evaluation(branch.left, update_injection<op>(tensor));
//			return expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
//		}
//	}
//	template<class core, int a, int b>
//	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//			if constexpr (expression_tree_evaluator<lv>::trivial_blas_eval && expression_tree_evaluator<rv>::trivial_blas_eval) {
//				expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
//				expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
//				return tensor.data();
//			}
//			//trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
//			else if constexpr (expression_tree_evaluator<lv>::trivial_blas_injection) {
//				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
//				auto right = expression_tree_evaluator<rv>::linear_evaluation(branch.right, update_injection<op>(tensor));
//
//				//if trivial blas evaluation right hand side, remove the right side branch (return left)
//				if constexpr(expression_tree_evaluator<rv>::trivial_blas_eval) {
//					return left;
//				} else {
//					//else return the adjusted types
//					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
//				}
//
//				//trivial injectable right hand side (prefer trivial injection right opposed to non-trivial)
//			} else if constexpr (expression_tree_evaluator<rv>::trivial_blas_injection) {
//				auto left = expression_tree_evaluator<lv>::linear_evaluation(branch.left, tensor);
//				auto right = expression_tree_evaluator<rv>::injection(branch.right, update_injection<op>(tensor));
//
//				//this branch will not get called, but this is to cover 'just-in-case' || showcase the parallels between lv/rv branches
//				if constexpr(expression_tree_evaluator<lv>::trivial_blas_eval) {
//					return right;
//				} else {
//					//this is what will be called
//					return binary_expression<std::decay_t<decltype(left)>, std::decay_t<decltype(right)>, op>(left, right);
//				}
//			}
//
//			//no trivial injections on either side, attempt to use nested injection //attempt to do non trivial injection left-side
//			//we do not need to use injection_update as the modifier will be evaluated when descending the tree (the branches are not removed)
//			else if constexpr (expression_tree_evaluator<lv>::non_trivial_blas_injection) {
//
//				auto left = expression_tree_evaluator<lv>::injection(branch.left, tensor);
//				//this should be changed to substitution evaluation when possible
//				auto right = branch.right; //rv
//				return binary_expression<std::decay_t<decltype(left)>, rv, op>(left, right);
//			}
//			//no trivial injections on either side, attempt to use nested injection //attempt to do non trivial injection right-side
//			else if constexpr (expression_tree_evaluator<rv>::non_trivial_blas_injection) {
//
//				//this should be changed to substitution evaluation when possible
//				auto left = branch.left; //lv
//				auto right = expression_tree_evaluator<rv>::injection(branch.right, tensor);
//				return binary_expression<lv, std::decay_t<decltype(right)>, op>(left, right);
//			} else {
//				throw std::invalid_argument("COMPLEX EVALUATOR FAILED TO DETECT AN INJECTION REPORT BUG");
//			}
//	}
//	static auto replacement(const binary_expression<lv,rv,op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			using branch_t = binary_expression<lv,rv,op>;
//			auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
//			return injection(branch, tmp);
//		} else {
//			return branch;
//		}
//	}
//	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			expression_tree_evaluator<lv>::destroy_temporaries(branch.left);
//			expression_tree_evaluator<rv>::destroy_temporaries(branch.right);
//		} else
//			return;
//	}
//
//};
//
//template<class lv, class rv, class op>
//struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
//	static constexpr bool trivial_blas_eval = true;
//	static constexpr bool trivial_blas_injection = true;
//	static constexpr bool non_trivial_blas_injection = true;
//	using branch_t = binary_expression<lv, rv, op>;
//
//	template<class core, int a, int b>
//	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//		branch.eval(tensor);
//		return tensor.data();
//	}
//	template<class core, int a, int b>
//	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
//		branch.eval(tensor);
//		return tensor.data();
//	}
//
//	//if no replacement is used yet, auto inject
//	static auto replacement(const binary_expression<lv, rv, op>& branch) {
//		auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
//		branch.eval(injection_wrapper<std::decay_t<decltype(tmp)>, 1, 0>(tmp));
//		return tmp;
//	}
//	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
//		if constexpr (non_trivial_blas_injection) {
//			expression_tree_evaluator<lv>::destroy_temporaries(branch.left);
//			expression_tree_evaluator<rv>::destroy_temporaries(branch.right);
//		} else
//			return;
//	}
//};
////----------------------------------------------------------------------------------------------------- implementation method
//template<class lv, class rv, class op>
//auto substitution_evaluate(binary_expression<lv, rv, op> expression) {
//	if constexpr (expression_tree_evaluator<binary_expression<lv, rv, op>>::non_trivial_blas_injection) {
//		return substitution_evaluate(expression_tree_evaluator<binary_expression<lv, rv, op>>::replacement(expression));
//	} else
//		return expression;
//}
////---------------------------------------------------------------------------------------------------------user methods
//template<class expression>
//void destroy_temporaries(expression expr) {
//	expression_tree_evaluator<expression>::destroy_temporaries(expr);
//}
//
//template<class lv, class rv>
//auto evaluate(binary_expression<lv, rv, oper::add_assign> expression) {
//	auto right = expression_tree_evaluator<rv>::linear_evaluation(expression.right, injection_wrapper<lv, 1, 1>(expression.left));
//	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::add_assign>(expression.left, right));
//}
//template<class lv, class rv>
//auto evaluate(binary_expression<lv, rv, oper::sub_assign> expression) {
//	auto right = expression_tree_evaluator<rv>::linear_evaluation(expression.right, injection_wrapper<lv, -1, 1>(expression.left));
//	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::sub_assign>(expression.left, right));
//}
//template<class lv, class rv>
//auto evaluate(binary_expression<lv, rv, oper::assign> expression) {
//	auto right = expression_tree_evaluator<rv>::injection(expression.right, injection_wrapper<lv, 1, 0>(expression.left));
//	return substitution_evaluate(binary_expression<lv, std::decay_t<decltype(right)>, oper::assign>(expression.left, right));
//}
//template<class lv, class rv>
//auto evaluate(binary_expression<lv, rv, oper::mul_assign> expression) {
//	return substitution_evaluate(expression);
//}
//template<class lv, class rv>
//auto evaluate(binary_expression<lv, rv, oper::div_assign> expression) {
//	return substitution_evaluate(expression);
//}
//
//
//
//
//}
//}
//}
//
//
//
//#endif /* PARSE_TREE_COMPLEX_EVALUATOR_H_ */
