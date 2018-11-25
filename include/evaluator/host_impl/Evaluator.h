/*
 * CPU_Evaluator.h
 *
 *  Created on: Oct 10, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
#define BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
namespace BC {
namespace evaluator {
namespace host_impl {

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

	template<class core_lib>
	struct Evaluator {
		template<int d, class expr_t>
		static void nd_evaluator(expr_t expr) {
			evaluator_impl<d>::impl(expr);
			 __BC_omp_bar__
		}
	};
}
}
}

#endif /* BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_ */
