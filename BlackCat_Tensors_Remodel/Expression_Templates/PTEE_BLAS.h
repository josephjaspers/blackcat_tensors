/*
 * PTEE_BLAS.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_BLAS_H_
#define PTEE_BLAS_H_

namespace BC {
namespace internal {
namespace tree {


template<class lv, class rv, class op>
struct expression_tree_evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
	static constexpr bool trivial_blas_eval = true;
	static constexpr bool trivial_blas_injection = true;
	static constexpr bool non_trivial_blas_injection = true;
	using branch_t = binary_expression<lv, rv, op>;

	template<class core, int a, int b>
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}
	template<class core, int a, int b>
	static auto injection(const binary_expression<lv, rv, op>& branch, injection_wrapper<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}

	//if no replacement is used yet, auto inject
	static auto replacement(const binary_expression<lv, rv, op>& branch) {
		auto tmp =  temporary<internal::Array<branch_t::DIMS(), scalar_of<branch_t>, mathlib_of<branch_t>>>(branch.inner_shape());
		branch.eval(injection_wrapper<std::decay_t<decltype(tmp)>, 1, 0>(tmp));
		return tmp;
	}
	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
		expression_tree_evaluator<lv>::destroy_temporaries(branch.left);
		expression_tree_evaluator<rv>::destroy_temporaries(branch.right);
	}
};

}
}
}



#endif /* PTEE_BLAS_H_ */
