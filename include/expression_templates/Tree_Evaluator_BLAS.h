/*
Author: Joseph F. Jaspers
Project: BlackCat_Tensors

    This file is part of BlackCat_Tensors.

    BlackCat_Tensors is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    BlackCat_Tensors is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with BlackCat_Tensors.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef PTEE_BLAS_H_
#define PTEE_BLAS_H_

#include "Tree_Evaluator_Common.h"
#include "Expression_Binary.h"

namespace BC {
namespace et     {
namespace tree {


template<class lv, class rv, class op>
struct evaluator<Binary_Expression<lv, rv, op>, std::enable_if_t<is_blas_func<op>()>> {
    static constexpr bool entirely_blas_expr = true;
    static constexpr bool partial_blas_expr = true;
    static constexpr bool nested_blas_expr = true;
    static constexpr bool requires_greedy_eval = true;


    using branch_t = Binary_Expression<lv, rv, op>;

    template<class core, int a, int b> __BChot__
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: linear_evaluation" << "alpha=" << a << "beta=" << b);

    	branch.eval(tensor);
        return tensor.data();
    }
    template<class core, int a, int b> __BChot__
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: injection");
        branch.eval(tensor);
        return tensor.data();
    }

    //if no replacement is used yet, auto inject
    __BChot__
    static auto temporary_injection(const Binary_Expression<lv, rv, op>& branch) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: temporary_injection");

    	using tmp_t = temporary<et::Array<branch_t::DIMS(), scalar_of<branch_t>, allocator_of<branch_t>>>;
        tmp_t tmp(branch.inner_shape());
        branch.eval(injector<tmp_t, 1, 0>(tmp));
        return tmp;
    }
    __BChot__
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: deallocate_temporaries");

        evaluator<lv>::deallocate_temporaries(branch.left);
        evaluator<rv>::deallocate_temporaries(branch.right);
    }
};

}
}
}



#endif /* PTEE_BLAS_H_ */
