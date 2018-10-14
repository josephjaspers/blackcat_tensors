/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

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

	void randomize(scalar_type lb=0, scalar_type ub=1)  {
		static_assert(derived::ITERATOR() == 0 || derived::ITERATOR() == 1,
				"randomize not available to non-continuous tensors");
		mathlib_type::randomize(as_derived().internal(), lb, ub);
	}
	void fill(scalar_type value)   { as_derived() = value; }
	void zero()                    { as_derived() = 0; 	   }

	template<class function>
	void for_each(function f) {
		auto for_each_expr = this->as_derived().un_expr(f);
		this->as_derived() = for_each_expr;
	}

	//transform is an alias for for_each
	template<class function>
	void transform(function f) {
		return for_each(f);
	}

	bool is_square() {
		return derived::DIMS() == 2 && as_derived().dimension(0) == as_derived().dimension(1);
	}
};

}
}




#endif /* TENSOR_FUNCTIONS_H_ */
