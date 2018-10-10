/*
 * PTEE_BLAS.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_BLAS_H_
#define PTEE_BLAS_H_

#include "Tree_Evaluator_Common.h"
#include "Expression_Binary.h"

namespace BC {
namespace internal {
namespace tree {


template<class lv, class rv, class op>
struct evaluator<Binary_Expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
	static constexpr bool trivial_blas_evaluation = true;
	static constexpr bool trivial_blas_injection = true;
	static constexpr bool non_trivial_blas_injection = true;
	using branch_t = Binary_Expression<lv, rv, op>;

	template<class core, int a, int b>
	static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}
	template<class core, int a, int b>
	static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
		branch.eval(tensor);
		return tensor.data();
	}

	//if no replacement is used yet, auto inject
	static auto replacement(const Binary_Expression<lv, rv, op>& branch) {
		auto tmp =  temporary<internal::Array<branch_t::DIMS(), scalar_of<branch_t>, mathlib_of<branch_t>>>(branch.inner_shape());
		branch.eval(injector<std::decay_t<decltype(tmp)>, 1, 0>(tmp));
		return tmp;
	}
	static void destroy_temporaries(const Binary_Expression<lv, rv, op>& branch) {
		evaluator<lv>::destroy_temporaries(branch.left);
		evaluator<rv>::destroy_temporaries(branch.right);
	}
};

}
}
}



#endif /* PTEE_BLAS_H_ */
