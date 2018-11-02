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
}



#endif /* TENSOR_RANDOM_H_ */
