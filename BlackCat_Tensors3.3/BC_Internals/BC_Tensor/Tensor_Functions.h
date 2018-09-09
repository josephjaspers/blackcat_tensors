/*
 * Tensor_Functions.h
 *
 *  Created on: Jun 5, 2018
 *      Author: joseph
 */

#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

namespace BC{
template<class internal> class Tensor_Base;

namespace module {
template<class derived> class Tensor_Functions;

template<class internal_t>
class Tensor_Functions<Tensor_Base<internal_t>> {
	template<class> friend class Tensor_Functions;

	using derived			= Tensor_Base<internal_t>;
	using scalar_type 		= typename internal_t::scalar_t;
	using mathlib_type 		= typename internal_t::mathlib_t;

	//Returns the class returned as its most derived member
	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }
public:

	void randomize(scalar_type lb, scalar_type ub)  { mathlib_type::randomize(as_derived().internal(), lb, ub); }
	void fill(scalar_type value) 					{ mathlib_type::fill(as_derived().internal(), value); }
	void zero() 									{ mathlib_type::zero(as_derived().internal()); }

};

}
}




#endif /* TENSOR_FUNCTIONS_H_ */
