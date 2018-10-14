/*
 * CPU_Evaluator.h
 *
 *  Created on: Oct 10, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_MATHLIBRARIES_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
#define BC_INTERNALS_BC_MATHLIBRARIES_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
namespace BC {
namespace cpu_evaluator {

template<class expression>
static void continuous_evaluate(expression expr) {
	__BC_omp_for__
	for (int i = 0; i < expr.size(); ++i) {
		expr[i];
	}
}

	template<int dim>
	struct evaluator {
		template<class expression, class... indexes>
		static void impl(expression expr, indexes... indicies) {
			__BC_omp_for__
			for (int i = 0; i < expr.dimension(dim); ++i) {
				evaluator<dim-1>::impl(expr, indicies..., i);
			}
		}
	};
	template<>
	struct evaluator<1> {
		template<class expression, class... indexes>
		static void impl(expression expr, indexes... indicies) {
			__BC_omp_for__
			for (int i = 0; i < expr.dimension(1); ++i) {
				expr(indicies..., i);
			}
		}
	};
	template<>
	struct evaluator<0> {
		template<class expression, class... indexes>
		static void impl(expression expr) {
			expr[0];
		}
	};
}

//--------------------------------
	template<class mathlib_core>
	struct CPU_Evaluator {
		template<int dim, class expression>
		static void nd_evaluator(expression expr) {
			if (dim == 0 || dim == 1)
				continuous_evaluate(expr);
			else
				cpu_evaluator::evaluator<dim>::impl(expr);
			__BC_omp_bar__
		}
	};


}


#endif /* BC_INTERNALS_BC_MATHLIBRARIES_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_ */
