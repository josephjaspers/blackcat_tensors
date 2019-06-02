/*
 * Tensor_CMath.h
 *
 *  Created on: Oct 30, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSOR_CMATH_H_
#define BLACKCAT_TENSOR_CMATH_H_

#include <cmath>

namespace BC {

//defines a function with a user defined implementation (not part of std::cmath.h)
#define BLACKCAT_BC_FUNCTOR_DEF(funcName, ...) 	 \
namespace module {										 \
														 \
	struct funcName {									 \
														 \
	  template<class value_type> BCINLINE    	     \
	  value_type operator () (value_type x) const { 	 \
		return __VA_ARGS__; 								 \
	  } 											     \
	  template<class value_type> BCINLINE    	     \
	  static value_type impl(value_type x) { 		     \
		return __VA_ARGS__; 								 \
	  }													 \
	};													 \
}													 	 \
														 \
template<class internal_t>                                       \
static auto funcName(const Tensor_Base<internal_t>& tensor) {    \
	return tensor.un_expr( module:: funcName () );               \
}

#define BLACKCAT_MATH_DEF(func) \
		BLACKCAT_BC_FUNCTOR_DEF(func, std::func(x))

BLACKCAT_MATH_DEF(abs)
BLACKCAT_MATH_DEF(acos)
BLACKCAT_MATH_DEF(acosh)
BLACKCAT_MATH_DEF(asin)
BLACKCAT_MATH_DEF(asinh)
BLACKCAT_MATH_DEF(atan)
BLACKCAT_MATH_DEF(atan2)
BLACKCAT_MATH_DEF(atanh)
BLACKCAT_MATH_DEF(cbrt)
BLACKCAT_MATH_DEF(ceil)
BLACKCAT_MATH_DEF(copysign)
BLACKCAT_MATH_DEF(cos)
BLACKCAT_MATH_DEF(cosh)
BLACKCAT_MATH_DEF(exp)
BLACKCAT_MATH_DEF(exp2)
BLACKCAT_MATH_DEF(expm1)
BLACKCAT_MATH_DEF(fabs)
BLACKCAT_MATH_DEF(fdim)
BLACKCAT_MATH_DEF(floor)
BLACKCAT_MATH_DEF(fma)
BLACKCAT_MATH_DEF(fmax)
BLACKCAT_MATH_DEF(fmin)
BLACKCAT_MATH_DEF(fmod)
BLACKCAT_MATH_DEF(frexp)
BLACKCAT_MATH_DEF(hypot)
BLACKCAT_MATH_DEF(ilogb)
BLACKCAT_MATH_DEF(ldexp)
BLACKCAT_MATH_DEF(llrint)
BLACKCAT_MATH_DEF(llround)
BLACKCAT_MATH_DEF(log)
BLACKCAT_MATH_DEF(log10)
BLACKCAT_MATH_DEF(log1p)
BLACKCAT_MATH_DEF(log2)
BLACKCAT_MATH_DEF(logb)
BLACKCAT_MATH_DEF(lrint)
BLACKCAT_MATH_DEF(lround)
BLACKCAT_MATH_DEF(modf)
BLACKCAT_MATH_DEF(nan)
BLACKCAT_MATH_DEF(nearbyint)
BLACKCAT_MATH_DEF(nextafter)
BLACKCAT_MATH_DEF(nexttoward)
BLACKCAT_MATH_DEF(pow)
BLACKCAT_MATH_DEF(remainder)
BLACKCAT_MATH_DEF(remquo)
BLACKCAT_MATH_DEF(rint)
BLACKCAT_MATH_DEF(round)
BLACKCAT_MATH_DEF(scalbln)
BLACKCAT_MATH_DEF(scalbn)
BLACKCAT_MATH_DEF(sin)
BLACKCAT_MATH_DEF(sinh)
BLACKCAT_MATH_DEF(sqrt)
BLACKCAT_MATH_DEF(tan)
BLACKCAT_MATH_DEF(tanh)
BLACKCAT_MATH_DEF(trunc)
BLACKCAT_MATH_DEF(isinf)
BLACKCAT_MATH_DEF(isnan)

BLACKCAT_BC_FUNCTOR_DEF(logistic, 1 / (1 + std::exp(-x)));
BLACKCAT_BC_FUNCTOR_DEF(dx_logistic, logistic::impl(x) * (1 - logistic::impl(x)));
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_logistic, (1 - x)*x);
BLACKCAT_BC_FUNCTOR_DEF(dx_tanh, 1 - std::pow(std::tanh(x), 2));
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_tanh, 1 - std::pow(x, 2));
BLACKCAT_BC_FUNCTOR_DEF(relu,std::max(0, x));
BLACKCAT_BC_FUNCTOR_DEF(dx_relu, x > 0 ? 1 : 0);
BLACKCAT_BC_FUNCTOR_DEF(cached_dx_relu, x > 0 ? 1 : 0); //same as dx_relu
BLACKCAT_BC_FUNCTOR_DEF(logical, x > 0 ? 1 : 0);
BLACKCAT_BC_FUNCTOR_DEF(sign, x > 0 ? 1 : x < 0 ? -1 : 0);

BLACKCAT_BC_FUNCTOR_DEF(pow2, std::pow(x, 2));

#undef BLACKCAT_BC_FUNCTOR_DEF
#undef BLACKCAT_MATH_DEF

//--------------------------------------------not actually cmath--------------------------------------//
template<class scalar>
struct norm {
    scalar min;
    scalar max;

    norm(scalar min_, scalar max_) : min(min_), max(max_) {}

    BCINLINE auto operator () (scalar v) const {
        return (v - min) / (max - min);
    }
};

template<class internal_t, class min_, class max_>
static auto normalize(const Tensor_Base<internal_t>& tensor, min_ min, max_ max) {
    using value_type = typename internal_t::value_type;
    return tensor.un_expr(norm<value_type>(value_type(min), value_type(max)));
}


}




#endif /* TENSOR_CMATH_H_ */
