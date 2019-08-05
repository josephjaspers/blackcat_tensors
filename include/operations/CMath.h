/*
 * Tensor_CMath.h
 *
 *  Created on: Oct 30, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSOR_CMATH_H_
#define BLACKCAT_TENSOR_CMATH_H_

#include <functional>
#include <cmath>

namespace BC {
namespace tensors {
template<class> class Tensor_Base;
}

namespace oper {
namespace cmath_functions {
//defines a function with a user defined implementation (not part of std::cmath.h)
#define BLACKCAT_FUNCTOR_DEF(funcName, instance_name, math_function, ...) 	\
														 	 	\
struct funcName {				 					 			\
	template<class value_type> BCINLINE    	     				\
	value_type operator () (const value_type& x) const { 	 	\
		return math_function; 									\
	} 											     			\
	template<class value_type> BCINLINE    	     				\
	static auto apply(const value_type& x) { 		    	 	\
		return math_function; 								 	\
	}															\
	template<class internal_t>									\
	auto operator() (const BC::tensors::Tensor_Base<internal_t>& tensor) {   \
		  return tensor.un_expr(funcName());               		\
	}															\
																\
	__VA_ARGS__													\
} instance_name; 										\
																\
																\
template<class T> auto surpress_unused_variable_warning_workaround(funcName)\
{return instance_name;}	   \
																\

#define DERIVATIVE_DEF(...) BLACKCAT_FUNCTOR_DEF(Derivative, dx, __VA_ARGS__)

#define DERIVATIVE_CACHED_DEF(...) BLACKCAT_FUNCTOR_DEF(Cached_Derivative, cached_dx, __VA_ARGS__)

#define BLACKCAT_MATH_DEF(funcName, instanceName, ...) \
		BLACKCAT_FUNCTOR_DEF(funcName, instanceName, std::instanceName(x), __VA_ARGS__)

//UTILITY 'just returns x'
BLACKCAT_FUNCTOR_DEF( Pass , pass, x, DERIVATIVE_DEF(1))

//COMMON
BLACKCAT_MATH_DEF( Exp , exp , DERIVATIVE_DEF(std::exp(x)))
BLACKCAT_MATH_DEF( Exp2 , exp2 )
BLACKCAT_MATH_DEF( Sqrt , sqrt, DERIVATIVE_DEF((std::pow(x, -1/2)/2)))

//Trig
BLACKCAT_MATH_DEF( Sin , sin, DERIVATIVE_DEF(std::cos(x)));
BLACKCAT_MATH_DEF( Cos , cos, DERIVATIVE_DEF(-std::sin(x)))
BLACKCAT_MATH_DEF( Tan , tan, DERIVATIVE_DEF(std::pow(1/std::cos(x), 2)))
BLACKCAT_FUNCTOR_DEF( Sec, sec, 1/std::cos(x) )

//Hyperbolic
BLACKCAT_MATH_DEF( Sinh , sinh, DERIVATIVE_DEF(std::cosh(x)))
BLACKCAT_MATH_DEF( Cosh , cosh, DERIVATIVE_DEF(std::sinh(x)))
BLACKCAT_MATH_DEF( Tanh , tanh, DERIVATIVE_DEF(1 - std::pow(std::tanh(x), 2))
										DERIVATIVE_CACHED_DEF(1 - std::pow(x,2)))
//Arc
BLACKCAT_MATH_DEF( Asin , asin, DERIVATIVE_DEF(1/std::sqrt(1-std::pow(x,2))))
BLACKCAT_MATH_DEF( Acos , acos, DERIVATIVE_DEF(-1/std::sqrt(1-std::pow(x,2))))
BLACKCAT_MATH_DEF( Atan , atan, DERIVATIVE_DEF(1/(1+std::pow(x,2))))
BLACKCAT_MATH_DEF( Atan2 , atan2 )

//Arc Hyperbolic
BLACKCAT_MATH_DEF( Asinh , asinh )
BLACKCAT_MATH_DEF( Acosh , acosh )
BLACKCAT_MATH_DEF( Atanh , atanh )

BLACKCAT_MATH_DEF( Abs , abs )
BLACKCAT_MATH_DEF( Cbrt , cbrt )
BLACKCAT_MATH_DEF( Ceil , ceil )
BLACKCAT_MATH_DEF( Copysign , copysign )
BLACKCAT_MATH_DEF( Expm1 , expm1 )
BLACKCAT_MATH_DEF( Fabs , fabs )
BLACKCAT_MATH_DEF( Fdim , fdim )
BLACKCAT_MATH_DEF( Floor , floor )
BLACKCAT_MATH_DEF( Fma , fma )
BLACKCAT_MATH_DEF( Fmax , fmax )
BLACKCAT_MATH_DEF( Fmin , fmin )
BLACKCAT_MATH_DEF( Fmod , fmod )
BLACKCAT_MATH_DEF( Frexp , frexp )
BLACKCAT_MATH_DEF( Hypot , hypot )
BLACKCAT_MATH_DEF( Ilogb , ilogb )
BLACKCAT_MATH_DEF( Ldexp , ldexp )
BLACKCAT_MATH_DEF( Llrint , llrint )
BLACKCAT_MATH_DEF( Llround , llround )
BLACKCAT_MATH_DEF( Log , log, DERIVATIVE_DEF(1/x))
BLACKCAT_MATH_DEF( Log10 , log10 )
BLACKCAT_MATH_DEF( Log1P , log1p )
BLACKCAT_MATH_DEF( Log2 , log2 )
BLACKCAT_MATH_DEF( Logb , logb )
BLACKCAT_MATH_DEF( Lrint , lrint )
BLACKCAT_MATH_DEF( Lround , lround )
BLACKCAT_MATH_DEF( Modf , modf )
BLACKCAT_MATH_DEF( Nan , nan )
BLACKCAT_MATH_DEF( Nearbyint , nearbyint )
BLACKCAT_MATH_DEF( Nextafter , nextafter )
BLACKCAT_MATH_DEF( Nexttoward , nexttoward )

struct Pow {

	template<class value_type, class Exp> BCINLINE
	value_type operator () (const value_type& x, Exp exp) const {
		return std::pow(x, exp);
	}
	template<class value_type, class Exp> BCINLINE
	static auto apply(const value_type& x, Exp exp) {
		return std::pow(x, exp);
	}
	template<class internal_t, class Exp>
	auto operator() (const BC::tensors::Tensor_Base<internal_t>& tensor, Exp exp) {
		struct FunctorPow {
			typename internal_t::value_type exp;
			 auto operator() (const typename internal_t::value_type value) const {
				return std::pow(value, exp);
			}
		};
		return tensor.un_expr(FunctorPow {exp});
	}
	template<class value_type, class Exp>
	static auto dx(const value_type&);

	template<class value_type, class Exp>
	static auto cached_dx(const value_type&);
} pow;

BLACKCAT_MATH_DEF( Remainder , remainder )
BLACKCAT_MATH_DEF( Remquo , remquo )
BLACKCAT_MATH_DEF( Rint , rint )
BLACKCAT_MATH_DEF( Round , round )
BLACKCAT_MATH_DEF( Scalbln , scalbln )
BLACKCAT_MATH_DEF( Scalbn , scalbn )
BLACKCAT_MATH_DEF( Trunc , trunc )
BLACKCAT_MATH_DEF( Isinf , isinf )
BLACKCAT_MATH_DEF( Isnan , isnan )

BLACKCAT_FUNCTOR_DEF(Pow2, pow2, (std::pow(x, 2)), DERIVATIVE_DEF(2));
BLACKCAT_FUNCTOR_DEF(Pow3, pow3, (std::pow(x, 3)), DERIVATIVE_DEF(3));

BLACKCAT_FUNCTOR_DEF(Logistic, logistic, (1 / (1 + std::exp(-x))),
		DERIVATIVE_DEF(Logistic::apply(x) * (1 - Logistic::apply(x)))
		DERIVATIVE_CACHED_DEF(x * (1 - x)));

BLACKCAT_FUNCTOR_DEF(Relu, relu, BC::traits::max(0, x), DERIVATIVE_DEF(x > 0 ? 1 : 0));
BLACKCAT_FUNCTOR_DEF(Logical, logical, x != 0 ? 1 : 0);




#undef BLACKCAT_FUNCTOR_DEF
#undef BLACKCAT_MATH_DEF

//--------------------------------------------not actually cmath--------------------------------------//
template<class scalar>
struct norm {
    scalar min;
    scalar max;

    BCINLINE norm(scalar min_, scalar max_) : min(min_), max(max_) {}
    BCINLINE auto operator () (scalar v) const {
        return (v - min) / (max - min);
    }
};

template<class internal_t, class min_, class max_>
static auto normalize(const BC::tensors::Tensor_Base<internal_t>& tensor, min_ min, max_ max) {
    using value_type = typename internal_t::value_type;
    return tensor.un_expr(norm<value_type>(value_type(min), value_type(max)));
}

} //end of ns cmath_functions
} //end of ns oper

using namespace BC::oper::cmath_functions;

} //end of ns BC



#endif /* TENSOR_CMATH_H_ */
