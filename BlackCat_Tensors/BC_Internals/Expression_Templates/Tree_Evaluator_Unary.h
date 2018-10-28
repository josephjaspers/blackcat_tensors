/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_UNARY_H_
#define PTEE_UNARY_H_

#include "Tree_Evaluator_Common.h"

namespace BC{
namespace internal {
namespace tree {

template<class array_t, class op>
struct evaluator<Unary_Expression<array_t, op>>
{
	static constexpr bool trivial_blas_evaluation = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = evaluator<array_t>::non_trivial_blas_injection;

	template<class core, int a, int b> __BChot__
	static auto linear_evaluation(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor) {
		return branch;
	}
	template<class core, int a, int b> __BChot__
	static auto injection(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor) {
		auto array =  evaluator<array_t>::injection(branch.array, tensor);
		using array_t_evaluated = std::decay_t<decltype(array)>;

		return Unary_Expression<array_t_evaluated, op>(array);
	}
	//keep calling replacement till all the replacements are needed

	struct trivial {
		__BChot__ static auto impl(const Unary_Expression<array_t, op>& branch) {
			using branch_t = Unary_Expression<array_t, op>;
			auto tmp =  temporary<internal::Array<branch_t::DIMS(), scalar_of<branch_t>, allocator_of<branch_t>>>(branch.inner_shape());
			return injection(branch, tmp);
		}
	};
	struct nontrivial {
		__BChot__ static auto impl(const Unary_Expression<array_t, op>& branch) {
			return branch;
		};
	};
	__BChot__ static auto replacement(const Unary_Expression<array_t, op>& branch) {

		using function = std::conditional_t<non_trivial_blas_injection, trivial, nontrivial>;
		return function::impl(branch);
	}
	__BChot__ static void deallocate_temporaries(const Unary_Expression<array_t, op>& branch) {
		evaluator<array_t>::deallocate_temporaries(branch.array);
	}
};


}
}
}



#endif /* PTEE_UNARY_H_ */
