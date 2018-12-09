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

namespace BC {
namespace et {
namespace tree {

template<class array_t, class op>
struct evaluator<Unary_Expression<array_t, op>>
{
    static constexpr bool entirely_blas_expr 	= false;
    static constexpr bool partial_blas_expr 	= false;
    static constexpr bool nested_blas_expr 		= evaluator<array_t>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= evaluator<array_t>::requires_greedy_eval;

    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor) {
        return branch;
    }
    template<class core, int a, int b> __BChot__
    static auto injection(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor) {
        auto array =  evaluator<array_t>::injection(branch.array, tensor);
        return Unary_Expression<decltype(array), op>(array);
    }

    __BChot__ static auto temporary_injection(const Unary_Expression<array_t, op>& branch) {

    	auto expr = evaluator<array_t>::temporary_injection(branch.array);
    	return Unary_Expression<std::decay_t<decltype(expr)>, op>(expr);

    }
    __BChot__ static void deallocate_temporaries(const Unary_Expression<array_t, op>& branch) {
        evaluator<array_t>::deallocate_temporaries(branch.array);
    }
};


}
}
}



#endif /* PTEE_UNARY_H_ */
