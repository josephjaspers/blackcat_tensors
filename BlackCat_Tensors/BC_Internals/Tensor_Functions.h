/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "Expression_Templates/Operations/Unary.h"

#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

namespace BC{
template<class internal> class Tensor_Base;

namespace module {
template<class derived> class Tensor_Functions;

template<class internal_t>
class Tensor_Functions<Tensor_Base<internal_t>> {
	template<class> friend class Tensor_Functions;

	using derived	    = Tensor_Base<internal_t>;
	using scalar_t 		= typename internal_t::scalar_t;
	using allocator_t     = typename internal_t::allocator_t;

	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }
public:

//-------------------------in-place functions-------------------------//
	void randomize(scalar_t lb=0, scalar_t ub=1)  {
		static_assert(derived::ITERATOR() == 0 || derived::ITERATOR() == 1,
				"randomize not available to non-continuous tensors");
		allocator_t::randomize(as_derived().internal(), lb, ub);
	}
	void fill(scalar_t value)   { as_derived() = value; }
	void zero()                 { as_derived() = scalar_t(0); }
	void ones()					{ as_derived() = scalar_t(1); }

	template<class function>
	void for_each(function f) {
		auto for_each_expr = this->as_derived().un_expr(f);
		this->as_derived() = for_each_expr;
	}
}; //end_of class 'Tensor_Functions'
}  //end_of namespace 'module'

//--------------------------lazy functions-------------------------//
template<class internal_t, class min_, class max_>
static auto normalize(const Tensor_Base<internal_t>& tensor, min_ min, max_ max) {
	using scalar_t = typename internal_t::scalar_t;
	return tensor.un_expr(internal::oper::norm<scalar_t>(scalar_t(min), scalar_t(max)));
}

#define BLACKCAT_LAZY_EXPR(func)                                      \
	template<class internal_t>                                    \
		static auto func(const Tensor_Base<internal_t>& tensor) {	  \
			return tensor.un_expr(internal::oper::func());	          \
		}
BLACKCAT_LAZY_EXPR(logistic)
BLACKCAT_LAZY_EXPR(dx_logistic)
BLACKCAT_LAZY_EXPR(cached_dx_logistic)
BLACKCAT_LAZY_EXPR(dx_tanh)
BLACKCAT_LAZY_EXPR(cached_dx_tanh)
BLACKCAT_LAZY_EXPR(relu)
BLACKCAT_LAZY_EXPR(dx_relu)
BLACKCAT_LAZY_EXPR(cached_dx_relu)

BLACKCAT_LAZY_EXPR(abs)
BLACKCAT_LAZY_EXPR(acos)
BLACKCAT_LAZY_EXPR(acosh)
BLACKCAT_LAZY_EXPR(sin)
BLACKCAT_LAZY_EXPR(asin)
BLACKCAT_LAZY_EXPR(asinh)
BLACKCAT_LAZY_EXPR(atan)
BLACKCAT_LAZY_EXPR(atanh)
BLACKCAT_LAZY_EXPR(cbrt)
BLACKCAT_LAZY_EXPR(ceil)
BLACKCAT_LAZY_EXPR(cos)
BLACKCAT_LAZY_EXPR(cosh)
BLACKCAT_LAZY_EXPR(exp)
BLACKCAT_LAZY_EXPR(exp2)
BLACKCAT_LAZY_EXPR(fabs)
BLACKCAT_LAZY_EXPR(floor)
BLACKCAT_LAZY_EXPR(fma)
BLACKCAT_LAZY_EXPR(isinf)
BLACKCAT_LAZY_EXPR(isnan)
BLACKCAT_LAZY_EXPR(log)
BLACKCAT_LAZY_EXPR(log2)
BLACKCAT_LAZY_EXPR(lrint)
BLACKCAT_LAZY_EXPR(lround)
BLACKCAT_LAZY_EXPR(modf)
BLACKCAT_LAZY_EXPR(sqrt)
BLACKCAT_LAZY_EXPR(tan)
BLACKCAT_LAZY_EXPR(tanh)

}






#endif /* TENSOR_FUNCTIONS_H_ */
