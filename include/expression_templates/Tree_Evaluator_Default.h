/*
 * Tree_Evaluator_Base.h
 *
 *  Created on: Dec 2, 2018
 *      Author: joseph
 */

#ifndef TREE_EVALUATOR_BASE_H_
#define TREE_EVALUATOR_BASE_H_

#include "Tree_Evaluator_Common.h"

namespace BC {
namespace et     {
namespace tree {

//disable default implementation (specialize for each type to ensure correct compilation)

template<class T>
struct evaluator_default
{
	/*
	 * entirely_blas_expr -- if we may replace this branch entirely with a temporary/cache
	 * partial_blas_expr  -- if part of this branch contains a replaceable branch nested inside it
	 * nested_blas_expr   -- if a replaceable branch is inside a function (+=/-= won't work but basic assign = can work)
	 */

    static constexpr bool entirely_blas_expr = false;			//An expression of all +/- operands and BLAS calls				IE w*x + y*z
    static constexpr bool partial_blas_expr = false;			//An expression of element-wise +/- operations and BLAS calls	IE w + x*y
    static constexpr bool nested_blas_expr  = false;			//An expression containing a BLAS expression nested in a unary_functor IE abs(w * x)
    static constexpr bool requires_greedy_eval = false;			//Basic check if any BLAS call exists at all

    template<class core, int a, int b>
    static auto linear_evaluation(const T& branch, injector<core, a, b> tensor) {
        return branch;
    }

    template<class core, int a, int b>
    static auto injection(const T& branch, injector<core, a, b> tensor) {
        return branch;
    }

    static auto temporary_injection(const T& branch) {
        return branch;
    }
    static void deallocate_temporaries(const T& tmp) {
        return;
    }
};


}
}
}





#endif /* TREE_EVALUATOR_BASE_H_ */
