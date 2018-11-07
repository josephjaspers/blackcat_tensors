/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

#include <algorithm>

namespace BC{
class CPU;
class GPU;
template<class internal> class Tensor_Base;

namespace module {
template<class derived> class Tensor_Functions;

enum functional_library {
	std = 0,
	thrust = 1
};

template<class internal_t>
class Tensor_Functions<Tensor_Base<internal_t>> {
	template<class> friend class Tensor_Functions;

	using derived	    = Tensor_Base<internal_t>;
	using scalar_t 		= typename internal_t::scalar_t;
	using allocator_t   = typename internal_t::allocator_t;

	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }


public:

//-------------------------in-place functions-------------------------//
	//TODO enable support for multi-dimensional randomize

	void fill(scalar_t value)   { as_derived() = value; }
	void zero()                 { as_derived() = scalar_t(0); }
	void ones()					{ as_derived() = scalar_t(1); }

   	void randomize(scalar_t lb=0, scalar_t ub=1)  {
   		static_assert(internal_t::ITERATOR() == 0 || internal_t::ITERATOR() == 1,
   				"randomize not available to non-continuous tensors");
   		allocator_t::randomize(this->as_derived().internal(), lb, ub);
   	}

}; //end_of class 'Tensor_Functions'


}  //end_of namespace 'module'


//--------------------------lazy functions-------------------------//
//TODO Add more loss functions, move loss functions to seperate module

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






#endif /* TENSOR_FUNCTIONS_H_ */
