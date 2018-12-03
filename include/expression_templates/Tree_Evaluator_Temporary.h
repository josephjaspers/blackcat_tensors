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
namespace et     {
namespace tree {


template<class core>
struct evaluator<temporary<core>>
{
    static constexpr bool entirely_blas_expr = false;
    static constexpr bool partial_blas_expr = false;
    static constexpr bool nested_blas_expr = false;
    static constexpr bool requires_greedy_eval = false;

    template<int a, int b> __BChot__
    static auto linear_evaluation(const temporary<core>& branch, injector<core, a, b> tensor) {
        return branch;
    }
    template<int a, int b> __BChot__
    static auto injection(const temporary<core>& branch, injector<core, a, b> tensor) {
        return branch;
    }
    __BChot__ static auto replacement(const temporary<core>& branch) {
        return branch;
    }
    __BChot__ static void deallocate_temporaries(temporary<core> tmp) {
        tmp.deallocate();
    }
};


}
}
}



#endif /* PTEE_TEMPORARY_H_ */
