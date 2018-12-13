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
#include "algorithms/Algorithms.h"
namespace BC {

#define BC_TENSOR_ALGORITHM_DEF(function)\
\
    template<class iter_begin_, class iter_end_, class... args>\
    static auto function (iter_begin_ begin_, iter_end_ end_, args... params) {\
        using tensor_t = typename iter_end_::tensor_t;\
        using allocator_t = typename tensor_t::allocator_t;\
        using system_tag  = typename allocator_t::system_tag;\
        using implementation = typename BC::algorithms::template implementation<system_tag>;\
\
        return implementation:: function (begin_, end_, params...);\
    }

namespace alg {
//---------------------------non-modifying sequences---------------------------//
BC_TENSOR_ALGORITHM_DEF(all_of)
BC_TENSOR_ALGORITHM_DEF(any_of)
BC_TENSOR_ALGORITHM_DEF(none_of)
BC_TENSOR_ALGORITHM_DEF(for_each)
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(for_each_n) )
BC_TENSOR_ALGORITHM_DEF(count)
BC_TENSOR_ALGORITHM_DEF(count_if)
BC_TENSOR_ALGORITHM_DEF(find)
BC_TENSOR_ALGORITHM_DEF(find_if)
BC_TENSOR_ALGORITHM_DEF(find_if_not)
BC_TENSOR_ALGORITHM_DEF(find_end)
BC_TENSOR_ALGORITHM_DEF(find_first_of)
BC_TENSOR_ALGORITHM_DEF(adjacent_find)
BC_TENSOR_ALGORITHM_DEF(search)
BC_TENSOR_ALGORITHM_DEF(search_n)
//modifying sequences
BC_TENSOR_ALGORITHM_DEF(copy)
BC_TENSOR_ALGORITHM_DEF(copy_if)
BC_TENSOR_ALGORITHM_DEF(copy_n)
BC_TENSOR_ALGORITHM_DEF(copy_backward)
BC_TENSOR_ALGORITHM_DEF(move)
BC_TENSOR_ALGORITHM_DEF(move_backward)
BC_TENSOR_ALGORITHM_DEF(fill)
BC_TENSOR_ALGORITHM_DEF(fill_n)
BC_TENSOR_ALGORITHM_DEF(transform)
BC_TENSOR_ALGORITHM_DEF(generate)
BC_TENSOR_ALGORITHM_DEF(generate_n)
BC_TENSOR_ALGORITHM_DEF(replace)
BC_TENSOR_ALGORITHM_DEF(replace_if)
BC_TENSOR_ALGORITHM_DEF(replace_copy)
BC_TENSOR_ALGORITHM_DEF(replace_copy_if)
BC_TENSOR_ALGORITHM_DEF(swap)
BC_TENSOR_ALGORITHM_DEF(swap_ranges)
BC_TENSOR_ALGORITHM_DEF(iter_swap)
BC_TENSOR_ALGORITHM_DEF(reverse)
BC_TENSOR_ALGORITHM_DEF(reverse_copy)
BC_TENSOR_ALGORITHM_DEF(rotate)
BC_TENSOR_ALGORITHM_DEF(rotate_copy)

BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(shift_left))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(shift_right))
BC_TENSOR_ALGORITHM_DEF(random_shuffle)
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(sample))
//--------------------------- sorting ---------------------------//
BC_TENSOR_ALGORITHM_DEF(is_sorted)
BC_TENSOR_ALGORITHM_DEF(is_sorted_until)
BC_TENSOR_ALGORITHM_DEF(sort)
BC_TENSOR_ALGORITHM_DEF(partial_sort)
BC_TENSOR_ALGORITHM_DEF(partial_sort_copy)
BC_TENSOR_ALGORITHM_DEF(stable_sort)
BC_TENSOR_ALGORITHM_DEF(nth_element)
BC_TENSOR_ALGORITHM_DEF(lower_bound)
BC_TENSOR_ALGORITHM_DEF(upper_bound)
BC_TENSOR_ALGORITHM_DEF(binary_search)
BC_TENSOR_ALGORITHM_DEF(equal_range)

//--------------------------- min/max ---------------------------//
BC_TENSOR_ALGORITHM_DEF(max)
BC_TENSOR_ALGORITHM_DEF(max_element)
BC_TENSOR_ALGORITHM_DEF(min)
BC_TENSOR_ALGORITHM_DEF(min_element)
BC_TENSOR_ALGORITHM_DEF(minmax)
BC_TENSOR_ALGORITHM_DEF(minmax_element)
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(clamp))
BC_TENSOR_ALGORITHM_DEF(equal)
BC_TENSOR_ALGORITHM_DEF(lexicographical_compare)

//--------------------------- numeric (mostly undefined) ---------------------------//
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(iota))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(accumulate))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(inner_product))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(adjacent_difference))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(partial_sum))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(reduce))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(exclusive_scan))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(inclusive_scan))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(transform_reduce))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(transform_exclusive_scan))
BC_DEF_IF_CPP17(BC_TENSOR_ALGORITHM_DEF(transform_inclusive_scan))
BC_TENSOR_ALGORITHM_DEF(qsort)
BC_TENSOR_ALGORITHM_DEF(bsearch)

}//end of namespace alg

using namespace alg;
template<class internal> class Tensor_Base;

namespace module {

template<class derived> class Tensor_Algorithm;

template<class internal_t>
class Tensor_Algorithm<Tensor_Base<internal_t>> {
    template<class> friend class Tensor_Algorithm;

    using derived        = Tensor_Base<internal_t>;
    using scalar_t         = typename internal_t::scalar_t;
    using allocator_t   = typename internal_t::allocator_t;

     const derived& as_derived() const { return static_cast<const derived&>(*this);  }
           derived& as_derived()       { return static_cast<      derived&>(*this); }

    auto begin_() { return this->as_derived().begin(); }
    auto end_() { return this->as_derived().end(); }
    auto cbegin_() const { return this->as_derived().begin(); }
    auto cend_() const { return this->as_derived().end(); }

public:

    void fill(scalar_t value)   { BC::fill(as_derived().begin_(), as_derived().end_(), value);}
    void zero()                 { fill(0); }
    void ones()                 { fill(1); }

    template<class function>
    void for_each(function func) {
    	as_derived() = as_derived().un_expr(func);
    }
    template<class function>
	void for_each(function func) const {
    	as_derived() = as_derived().un_expr(func);
    }

    void sort() {
    	BC::alg::sort(this->begin_(), this->end_());
    }

    void rand(scalar_t lb=0, scalar_t ub=1) {
    	randomize(lb, ub);
	}

   void randomize(scalar_t lb=0, scalar_t ub=1)  {
	   static_assert(internal_t::ITERATOR() == 0 || internal_t::ITERATOR() == 1,
			   	   	   "randomize not available to non-continuous tensors");

	   using impl = random::implementation<typename allocator_t::system_tag>;
	   impl::randomize(this->as_derived().internal(), lb, ub);
   }

   scalar_t max() const {
	   auto max_index = BC::alg::max_element(this->cbegin_(), this->cend_());
	   return allocator_t::extract(this->as_derived().memptr(), max_index);
   }
   scalar_t min() const {
	   auto min_index = BC::alg::min_element(this->cbegin_(), this->cend_());
	   return allocator_t::extract(this->as_derived().memptr(), min_index);
   }

   //if bool flip the output of sum to an integer
   using sum_t = std::conditional_t<std::is_same<bool, scalar_t>::value, int, scalar_t>;

   BC_DEF_IF_CPP17(
		sum_t sum() const {
		   return BC::accumulate(this->cbegin_(), this->cend_(), sum_t(0));
	   }
   )

   BC_DEF_IF_CPP17(
   scalar_t prod() const {
	   return BC::accumulate(this->cbegin_(), this->cend_(), scalar_t(1), BC::et::oper::mul());
   }
   )

#define BC_TENSOR_ALGORITHM_MEMBER_DEF(function)\
  template<class... args>\
  auto function (args... params) const {\
	  return BC::alg:: function (this->cbegin_(), this->cend_(), params...);\
  }

   BC_TENSOR_ALGORITHM_MEMBER_DEF(all_of)
   BC_TENSOR_ALGORITHM_MEMBER_DEF(any_of)
   BC_TENSOR_ALGORITHM_MEMBER_DEF(none_of)
   BC_TENSOR_ALGORITHM_MEMBER_DEF(max_element)
   BC_TENSOR_ALGORITHM_MEMBER_DEF(min_element)

}; //end_of class 'Tensor_Functions'

}  //end_of namespace 'module'

template<class internal_t>
static bool all(const Tensor_Base<internal_t>& tensor) {
	constexpr int dims = internal_t::DIMS();
	using scalar_t = typename internal_t::scalar_t;
	using allocator_t  = typename allocator::template implementation<typename internal_t::system_tag, scalar_t>;

	Tensor_Base<BC::et::Array<dims, bool, allocator_t>> bool_tensor(tensor.inner_shape());
	bool_tensor = logical(tensor);

	return bool_tensor.sum() == bool_tensor.size();
}
template<class internal_t>
static bool any(const Tensor_Base<internal_t>& tensor) {
	constexpr int dims = internal_t::DIMS();
	using scalar_t = typename internal_t::scalar_t;
	using allocator_t  = typename allocator::template implementation<typename internal_t::system_tag, scalar_t>;

	Tensor_Base<BC::et::Array<dims, bool, allocator_t>>
	bool_tensor(tensor.inner_shape());
	bool_tensor = logical(tensor);

	return bool_tensor.sum()!= 0;
}






}  //end_of namespace 'BC'

#endif /* TENSOR_FUNCTIONS_H_ */
