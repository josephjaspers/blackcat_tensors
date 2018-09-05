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
struct evaluator<binary_expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
	static constexpr bool trivial_blas_feature_detector = true;
	static constexpr bool trivial_blas_injection = true;
	static constexpr bool non_trivial_blas_injection = true;
	using branch_t = binary_expression<lv, rv, op>;

	template<class core, int a, int b>
	static auto linear_evaluation(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}
	template<class core, int a, int b>
	static auto injection(const binary_expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}

	//if no replacement is used yet, auto inject
	static auto replacement(const binary_expression<lv, rv, op>& branch) {
		auto tmp =  temporary<internal::Array<branch_t::DIMS(), scalar_of<branch_t>, mathlib_of<branch_t>>>(branch.inner_shape());
		branch.eval(injector<std::decay_t<decltype(tmp)>, 1, 0>(tmp));
		return tmp;
	}
	static void destroy_temporaries(const binary_expression<lv, rv, op>& branch) {
		evaluator<lv>::destroy_temporaries(branch.left);
		evaluator<rv>::destroy_temporaries(branch.right);
	}
};

}
}
}



#endif /* PTEE_BLAS_H_ */
