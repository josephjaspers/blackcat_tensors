/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Tree_Evaluator_Common.h"
#include "Array_Base.h"

namespace BC {
namespace et     {
namespace tree {

//disable default implementation (specialize for each type to ensure correct compilation)

template<class T>
struct evaluator<T, std::enable_if_t<is_array<T>()>>
{
    static constexpr bool entirely_blas_expr = false;
    static constexpr bool partial_blas_expr = false;
    static constexpr bool nested_blas_expr  = false;
    static constexpr bool requires_greedy_eval = false;

    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const T& branch, injector<core, a, b> tensor) {
        return branch;
    }
    template<class core, int a, int b> __BChot__
    static auto injection(const T& branch, injector<core, a, b> tensor) {
        return branch;
    }

    __BChot__ static auto replacement(const T& branch) {
        return branch;
    }
    __BChot__ static void deallocate_temporaries(const T& tmp) {
        return;
    }
};


}
}
}



#endif /* PTE_ARRAY_H_ */
