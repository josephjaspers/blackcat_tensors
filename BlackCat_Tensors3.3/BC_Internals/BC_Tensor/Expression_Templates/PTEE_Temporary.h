/*
 * PTEE_Temporary.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTEE_TEMPORARY_H_
#define PTEE_TEMPORARY_H_

namespace BC {
namespace internal {
namespace tree {


template<class core>
struct expression_tree_evaluator<temporary<core>>
{
	static constexpr bool trivial_blas_eval = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = false;

	template<int a, int b>
	static auto linear_evaluation(const temporary<core>& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	template<int a, int b>
	static auto injection(const temporary<core>& branch, injection_wrapper<core, a, b> tensor) {
		return branch;
	}
	static auto replacement(const temporary<core>& branch) {
		return branch;
	}
	static void destroy_temporaries(temporary<core> tmp) {
		tmp.destroy();
	}
};


}
}
}



#endif /* PTEE_TEMPORARY_H_ */
