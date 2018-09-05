/*
 * Tensor_Functions.h
 *
 *  Created on: Jun 5, 2018
 *      Author: joseph
 */

#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

namespace BC{
namespace module {

template<class derived>
class Tensor_Functions {
	template<class> friend class Tensor_Functions;

	using functor_type 		= functor_of<derived>;
	using scalar_type 		= scalar_of<derived>;
	using mathlib_type 		= mathlib_of<derived>;

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
