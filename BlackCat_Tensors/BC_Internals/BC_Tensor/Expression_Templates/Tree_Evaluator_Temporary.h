/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_TEMPORARY_H_
#define PTEE_TEMPORARY_H_

namespace BC {
namespace internal {
namespace tree {


template<class core>
struct evaluator<temporary<core>>
{
	static constexpr bool trivial_blas_evaluation = false;
	static constexpr bool trivial_blas_injection = false;
	static constexpr bool non_trivial_blas_injection = false;

	template<int a, int b>
	static auto linear_evaluation(const temporary<core>& branch, injector<core, a, b> tensor) {
		return branch;
	}
	template<int a, int b>
	static auto injection(const temporary<core>& branch, injector<core, a, b> tensor) {
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
