/*
 * PTE_Array.h
 *
 *  Created on: Jun 21, 2018
 *      Author: joseph
 */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Tree_Evaluator_Common.h"
#include "Array_Base.h"

namespace BC {
namespace internal {
namespace tree {

//disable default implementation (specialize for each type to ensure correct compilation)
template<class T, class enabler = void>
struct evaluator;

template<class T>
struct evaluator<T, std::enable_if_t<is_array<T>()>>
{
	static constexpr bool trivial_blas_evaluation = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = false;

	template<class core, int a, int b>
	static auto linear_evaluation(const T& branch, injector<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b>
	static auto injection(const T& branch, injector<core, a, b> tensor) {
		return branch;
	}
	static auto replacement(const T& branch) {
		return branch;
	}
	static void destroy_temporaries(const T& tmp) {
		return;
	}
};


}
}
}



#endif /* PTE_ARRAY_H_ */
