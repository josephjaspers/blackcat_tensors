/*
 * Tensor_Functions.h
 *
 *  Created on: Jun 5, 2018
 *      Author: joseph
 */

#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

namespace BC{
namespace Base {

template<class derived>
class Tensor_Functions {
	template<class> friend class Tensor_Functions;
	template<class pderiv, class functor> using impl 	= typename operationImpl::expression_determiner<derived>::template impl<pderiv, functor>;
	template<class pderiv> 				  using dp_impl	= typename operationImpl::expression_determiner<derived>::template dp_impl<pderiv>;

	using functor_type 		= _functor<derived>;
	using scalar_type 		= _scalar<derived>;
	using mathlib_type 		= _mathlib<derived>;

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
