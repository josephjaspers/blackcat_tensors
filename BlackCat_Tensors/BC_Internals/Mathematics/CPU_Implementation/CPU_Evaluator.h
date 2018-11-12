/*
 * CPU_Evaluator.h
 *
 *  Created on: Oct 10, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
#define BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_
namespace BC {
namespace cpu_evaluator {

    template<int dim>
    struct evaluator {
        template<class expression, class... indexes>
        static void impl(expression expr, indexes... indicies) {
            __BC_omp_for__
            for (int i = 0; i < expr.dimension(dim-1); ++i) {
                evaluator<dim-1>::impl(expr, indicies..., i);
            }
        }
    };
    template<>
    struct evaluator<1> {
        template<class expression, class... indexes>
        static void impl(expression expr, indexes... indicies) {
            __BC_omp_for__
            for (int i = 0; i < expr.dimension(0); ++i) {
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
    template<class allocator_core>
    struct CPU_Evaluator {
    	    //-------------------------------1d eval/copy ---------------------------------//

    	    struct n0 {
    	        template<class T>
    	        static void eval(T to) {
    	 __BC_omp_for__
    	            for (int i = 0; i < to.size(); ++i) {
    	                to[i];
    	            }
    	        }
    	    };

    	    //-------------------------------2d eval/copy ---------------------------------//
    	    struct n2 {
    	        template<class T>
    	        static void eval(T to) {
    	 __BC_omp_for__
    	            for (int n = 0; n < to.dimension(1); ++n)
    	                for (int m = 0; m < to.dimension(0); ++m)
    	                    to(m, n);
    	        }
    	    };
    	    //-------------------------------3d eval/copy ---------------------------------//

    	    struct n3 {
    	        template<class T>
    	        static void eval(T to) {

    	 __BC_omp_for__

    	            for (int k = 0; k < to.dimension(2); ++k)
    	                for (int n = 0; n < to.dimension(1); ++n)
    	                    for (int m = 0; m < to.dimension(0); ++m)
    	                        to(m, n, k);
    	        }
    	    };
    	    //-------------------------------4d eval/copy ---------------------------------//4
    	    struct n4 {
    	        template<class T>
    	        static void eval(T to) {

    	 __BC_omp_for__
    	            for (int p = 0; p < to.dimension(3); ++p)
    	                for (int k = 0; k < to.dimension(2); ++k)
    	                    for (int n = 0; n < to.dimension(1); ++n)
    	                        for (int m = 0; m < to.dimension(0); ++m)
    	                            to(m, n, k, p);
    	        }
    	    };
    	    //-------------------------------5d eval/copy ---------------------------------//

    	    struct n5 {
    	        template<class T>
    	        static void eval(T to) {

    	 __BC_omp_for__
    	            for (int j = 0; j < to.dimension(4); ++j)
    	                for (int p = 0; p < to.dimension(3); ++p)
    	                    for (int k = 0; k < to.dimension(2); ++k)
    	                        for (int n = 0; n < to.dimension(1); ++n)
    	                            for (int m = 0; m < to.dimension(0); ++m)
    	                                to(m, n, k, p, j);
    	        }
    	    };
    	    //-------------------------------implementation ---------------------------------//

    	    template<int d>
    	    struct dimension {
    	        using run = std::conditional_t<(d <= 1), n0,
    	        std::conditional_t< d ==2, n2,
    	        std::conditional_t< d == 3, n3,
    	        std::conditional_t< d == 4, n4, n5>>>>;

    	        template<class T>
    	        static void eval(T to) {
    	            run::eval(to);
    	#ifndef __BC_parallel_section__
    	 __BC_omp_bar__
    	#endif
    	        }
    	    };

    	    template<int d, class expr_t>
    	    static void nd_evaluator(expr_t expr) {
    	        dimension<d>::eval(expr);
    	    }
//      need to fix this //FIXME RECURSIVE EVAUATOR FOR CPU AND GPU NEED TO WORK (SO FAR BUGGY)
//        template<class expression>
//        static void continuous_evaluate(expression expr) {
//            __BC_omp_for__
//            for (int i = 0; i < expr.size(); ++i) {
//                expr[i];
//            }
//        }
//
//        template<int dim, class expression>
//        static void nd_evaluator(expression expr) {
//            if (dim == 0 || dim == 1)
//                continuous_evaluate(expr);
//            else
//                cpu_evaluator::evaluator<dim>::impl(expr);
//            __BC_omp_bar__
//        }
    };


}


#endif /* BC_INTERNALS_BC_ALLOCATOR_CPU_IMPLEMENTATION_CPU_EVALUATOR_H_ */
