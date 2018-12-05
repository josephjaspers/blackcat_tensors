/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

namespace BC {
namespace evaluator {

template<int dim>
struct evaluator_impl {
    template<class expression, class... indexes>
    static void impl(expression expr, indexes... indicies) {
        __BC_omp_for__
        for (int i = 0; i < expr.dimension(dim-1); ++i) {
        	evaluator_impl<dim-1>::impl(expr, i, indicies...);
        }
    }
};
template<>
struct evaluator_impl<1> {
    template<class expression, class... indexes>
    static void impl(expression expr, indexes... indicies) {
        __BC_omp_for__
        for (int i = 0; i < expr.dimension(0); ++i) {
            expr(i, indicies...);
        }
    }
    template<class expression>
    static void impl(expression expr) {
        __BC_omp_for__
        for (int i = 0; i < expr.size(); ++i) {
            expr[i];
        }
    }
};
template<>
struct evaluator_impl<0> {
    template<class expression>
    static void impl(expression expr) {
        __BC_omp_for__
        for (int i = 0; i < expr.size(); ++i) {
            expr[i];
        }
    }
};


struct Host {

	template<int d, class expr_t>
	static void nd_evaluator(expr_t expr) {
		evaluator_impl<d>::impl(expr);
		 __BC_omp_bar__
	}

};
}
}
#endif
 /* MATHEMATICS_CPU_H_ */
