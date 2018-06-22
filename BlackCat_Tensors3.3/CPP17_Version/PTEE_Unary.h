/*
 * PTEE_Unary.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_UNARY_H_
#define PTEE_UNARY_H_

namespace BC{
namespace internal {
namespace tree {




template<class array_t, class op>
struct expression_tree_evaluator<unary_expression<array_t, op>>
{
	static constexpr bool trivial_blas_eval = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = expression_tree_evaluator<array_t>::non_trivial_blas_injection;

	template<class core, int a, int b>
	static auto linear_evaluation(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b>
	static auto injection(const unary_expression<array_t, op>& branch, injection_wrapper<core, a, b> tensor) {
		auto array =  expression_tree_evaluator<array_t>::injection(branch.array, tensor);
		using array_t_evaluated = std::decay_t<decltype(array)>;

		return unary_expression<array_t_evaluated, op>(array);
	}
	//keep calling replacement till all the replacements are needed

	struct trivial {
		static constexpr auto impl = [](auto& branch) {
			using branch_t = unary_expression<array_t, op>;
			auto tmp =  temporary<internal::Array<branch_t::DIMS(), _scalar<branch_t>, _mathlib<branch_t>>>(branch.inner_shape());
			return injection(branch, tmp);
		};
	};
	struct nontrivial {
		static constexpr auto impl = [] (auto& branch) {
			return branch;
		};
	};
	static auto replacement(const unary_expression<array_t, op>& branch) {

		using function = std::conditional_t<non_trivial_blas_injection, trivial, nontrivial>;
		return function::impl(branch);
	}
	static void destroy_temporaries(const unary_expression<array_t, op>& branch) {
		if constexpr (non_trivial_blas_injection) {
			expression_tree_evaluator<array_t>::destroy_temporaries(branch.array);
		} else
			return;
	}
};


}
}
}



#endif /* PTEE_UNARY_H_ */
