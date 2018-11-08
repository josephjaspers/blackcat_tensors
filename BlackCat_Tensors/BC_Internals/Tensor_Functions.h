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


#define BC_TENSOR_ALGORITHM_DEF(function) \
	template<class iter_begin, class iter_end, class functor>\
	static auto function (iter_begin begin, iter_end end, functor func) {\
		using tensor_t = typename iter_begin::tensor_t;\
		using mathlib_t = typename tensor_t::mathlib_t;\
\
		mathlib_t :: function (begin, end, func);\
	}

//	BC_CPU_ALGORITHM_DEF(all_of
//	BC_CPU_ALGORITHM_DEF(any_of
//	none_of
//	for_each
//	find
//	find_End
//	find_first_of
//	find_adjacent_find
//	count
//	mismatch
//	equal
//	is_permutation
//	search
//	search_n
//
	BC_TENSOR_ALGORITHM_DEF(all_of)
	BC_TENSOR_ALGORITHM_DEF(any_of)
	BC_TENSOR_ALGORITHM_DEF(none_of)
	BC_TENSOR_ALGORITHM_DEF(for_each)
	BC_TENSOR_ALGORITHM_DEF(find)
	//find if
	//find if not

	BC_TENSOR_ALGORITHM_DEF(find_end)
	BC_TENSOR_ALGORITHM_DEF(find_first_of)
	BC_TENSOR_ALGORITHM_DEF(adjacent_find)
	BC_TENSOR_ALGORITHM_DEF(count)
	//count if
	BC_TENSOR_ALGORITHM_DEF(mismatch)
	BC_TENSOR_ALGORITHM_DEF(equal)
	BC_TENSOR_ALGORITHM_DEF(is_permutation)
	BC_TENSOR_ALGORITHM_DEF(search)
	BC_TENSOR_ALGORITHM_DEF(search_n)


	//modifying-----------------------------------------

	BC_TENSOR_ALGORITHM_DEF(copy)
	BC_TENSOR_ALGORITHM_DEF(copy_n)
//	BC_TENSOR_ALGORITHM_DEF(adjacent_find)
	BC_TENSOR_ALGORITHM_DEF(copy_if)
	BC_TENSOR_ALGORITHM_DEF(copy_backward)
	BC_TENSOR_ALGORITHM_DEF(move)
	BC_TENSOR_ALGORITHM_DEF(move_backward)
	BC_TENSOR_ALGORITHM_DEF(swap)
	BC_TENSOR_ALGORITHM_DEF(swap_ranges)
	BC_TENSOR_ALGORITHM_DEF(iter_swap)
	BC_TENSOR_ALGORITHM_DEF(transform)
	BC_TENSOR_ALGORITHM_DEF(replace)
	BC_TENSOR_ALGORITHM_DEF(replace_if)
	BC_TENSOR_ALGORITHM_DEF(replace_copy)
	BC_TENSOR_ALGORITHM_DEF(replace_copy_if)
	BC_TENSOR_ALGORITHM_DEF(fill)

	BC_TENSOR_ALGORITHM_DEF(fill_n)
	BC_TENSOR_ALGORITHM_DEF(generate)
	BC_TENSOR_ALGORITHM_DEF(generate_n)
//	BC_TENSOR_ALGORITHM_DEF(remove)
//	BC_TENSOR_ALGORITHM_DEF(remove_if)

//	BC_TENSOR_ALGORITHM_DEF(remove_copy)
//	BC_TENSOR_ALGORITHM_DEF(remove_copy_if)
//	BC_TENSOR_ALGORITHM_DEF(unique)
//	BC_TENSOR_ALGORITHM_DEF(unique_copy)

}






#endif /* TENSOR_FUNCTIONS_H_ */
