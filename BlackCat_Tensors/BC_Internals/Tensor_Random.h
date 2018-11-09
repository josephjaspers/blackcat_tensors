/*
 * Tensor_Random.h
 *
 *  Created on: Oct 30, 2018
 *      Author: joseph
 */

#ifndef TENSOR_RANDOM_H_
#define TENSOR_RANDOM_H_
#include <type_traits>

namespace BC {
template<class> class Tensor_Base;

namespace module {

template<class derived>
class Tensor_Random;

template<class internal_t>
class Tensor_Random<Tensor_Base<internal_t>>{
//
//	using derived	    = Tensor_Base<internal_t>;
//	using scalar_t 		= typename internal_t::scalar_t;
//	using allocator_t   = typename internal_t::allocator_t;
//
//	static constexpr bool is_int_t    = std::is_integral<scalar_t>::value;
//	static constexpr bool is_float_t  = std::is_floating_point<scalar_t>::value;
//
//	using default_int_t = int;
//	using default_float_t = double;
//
//	using default_integer_t  = std::conditional_t<is_int_t, scalar_t, default_int_t>;
//	using default_floating_t = std::conditional_t<is_float_t, scalar_t, default_float_t>;
//
//	const derived& as_derived() const { return static_cast<derived&>(*this); }
//		  derived& as_derived() 	  { return static_cast<derived&>(*this); }
//
//public:
//
//
//		  Internals

};



}

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



#endif /* TENSOR_RANDOM_H_ */
