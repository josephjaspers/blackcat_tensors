/*
 * Tensor_CMath.h
 *
 *  Created on: Oct 30, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CMATH_H_
#define TENSOR_CMATH_H_

#include <cmath>
#include "expression_templates/Internal_Common.h"
#include "expression_templates/operations/Unary.h"

namespace BC {

//defines the functor object
#define BLACKCAT_MATH_DEF(func)                                     \
namespace functor {												    \
	struct func {                                                   \
		template<class scalar_t> __BCinline__                       \
		scalar_t operator () (scalar_t s) const { 					\
			return std::func(s); 									\
	}   															\
																	\
	  template<class scalar_t> __BCinline__                         \
	  static scalar_t impl(scalar_t s) {  return std::func(s); }    \
	};															    \
}																	\
                                                                    \
    template<class internal_t>                                      \
        static auto func(const Tensor_Base<internal_t>& tensor) {   \
            return tensor.un_expr( functor:: func () );             \
        }

BLACKCAT_MATH_DEF(abs)
BLACKCAT_MATH_DEF(acos)
BLACKCAT_MATH_DEF(acosh)
BLACKCAT_MATH_DEF(sin)
BLACKCAT_MATH_DEF(asin)
BLACKCAT_MATH_DEF(asinh)
BLACKCAT_MATH_DEF(atan)
BLACKCAT_MATH_DEF(atanh)
BLACKCAT_MATH_DEF(cbrt)
BLACKCAT_MATH_DEF(ceil)
BLACKCAT_MATH_DEF(cos)
BLACKCAT_MATH_DEF(cosh)
BLACKCAT_MATH_DEF(exp)
BLACKCAT_MATH_DEF(exp2)
BLACKCAT_MATH_DEF(fabs)
BLACKCAT_MATH_DEF(floor)
BLACKCAT_MATH_DEF(fma)
BLACKCAT_MATH_DEF(isinf)
BLACKCAT_MATH_DEF(isnan)
BLACKCAT_MATH_DEF(log)
BLACKCAT_MATH_DEF(log2)
BLACKCAT_MATH_DEF(lrint)
BLACKCAT_MATH_DEF(lround)
BLACKCAT_MATH_DEF(modf)
BLACKCAT_MATH_DEF(sqrt)
BLACKCAT_MATH_DEF(tan)
BLACKCAT_MATH_DEF(tanh)

//defines a function with a user defined implementation (not part of std::cmath.h)
#define BLACKCAT_BC_FUNCTOR_DEF(funcName, func_math) 	 \
namespace module {										 \
														 \
	struct funcName {									 \
														 \
	  template<class scalar_t> __BCinline__    			 \
	  scalar_t operator () (scalar_t x) const { 	     \
		return func_math; 								 \
	  } 											     \
	  template<class scalar_t> __BCinline__    			 \
	  static scalar_t impl(scalar_t x) { 				 \
		return func_math; 								 \
	  }													 \
	};													 \
}													 	 \
template<class internal_t>                                       \
static auto funcName(const Tensor_Base<internal_t>& tensor) {    \
	return tensor.un_expr( module:: funcName () );               \
}

BLACKCAT_BC_FUNCTOR_DEF(logistic, 1 / (1 + std::exp(-x)));
BLACKCAT_BC_FUNCTOR_DEF(dx_logistic, logistic::impl(x) * (1 - logistic::impl(x)));
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_logistic, x * (1 - x));
BLACKCAT_BC_FUNCTOR_DEF(dx_tanh, 1 - std::pow(std::tanh(x), 2));
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_tanh, 1 - std::pow(x, 2));
BLACKCAT_BC_FUNCTOR_DEF(relu,std::max(0, x));
BLACKCAT_BC_FUNCTOR_DEF(dx_relu, x > 0 ? 1 : 0);
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_relu, x > 0 ? 1 : 0); //same as dx_relu
BLACKCAT_BC_FUNCTOR_DEF(logical, x > 0 ? 1 : 0); //same as dx_relu


//--------------------------------------------not actually cmath--------------------------------------//
template<class scalar>
struct norm {
    scalar min;
    scalar max;

    norm(scalar min_, scalar max_) : min(min_), max(max_) {}

    __BCinline__ auto operator () (scalar v) const {
        return (v - min) / (max - min);
    }
};

template<class internal_t, class min_, class max_>
static auto normalize(const Tensor_Base<internal_t>& tensor, min_ min, max_ max) {
    using scalar_t = typename internal_t::scalar_t;
    return tensor.un_expr(norm<scalar_t>(scalar_t(min), scalar_t(max)));
}


}




#endif /* TENSOR_CMATH_H_ */
