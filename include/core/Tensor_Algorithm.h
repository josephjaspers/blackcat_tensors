/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BLACKCAT_TENSOR_ALGORITHMS_H_
#define BLACKCAT_TENSOR_ALGORITHMS_H_

#include <algorithm>
#include "algorithms/Algorithms.h"
namespace BC {

#define BC_TENSOR_ALGORITHM_DEF(function)\
\
    template<class iter_begin_, class iter_end_, class... args>\
    static auto function (iter_begin_ begin_, iter_end_ end_, args... params) {\
        using tensor_t = typename iter_end_::tensor_t;\
        using allocator_t = typename tensor_t::allocator_t;\
        using system_tag  = typename BC::allocator_traits<allocator_t>::system_tag;\
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

#undef BC_TENSOR_ALGORITHM_DEF
}//end of namespace alg

using namespace alg;
template<class internal> class Tensor_Base;

namespace module {

template<class derived> class Tensor_Algorithm;

template<class internal_t>
class Tensor_Algorithm<Tensor_Base<internal_t>> {
    template<class> friend class Tensor_Algorithm;

    using derived       = Tensor_Base<internal_t>;
    using value_type    = typename internal_t::value_type;
    using allocator_t   = typename internal_t::allocator_t;
    using system_tag    = typename BC::allocator_traits<allocator_t>::system_tag;
    using utility_t     = typename utility::implementation<system_tag>;


     const derived& as_derived() const { return static_cast<const derived&>(*this);  }
           derived& as_derived()       { return static_cast<      derived&>(*this); }

    auto begin_() { return this->as_derived().begin(); }
    auto end_() { return this->as_derived().end(); }
    auto cbegin_() const { return this->as_derived().begin(); }
    auto cend_() const { return this->as_derived().end(); }

public:

    void fill(value_type value)   { BC::fill(as_derived().begin_(), as_derived().end_(), value);}
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

    void rand(value_type lb=0, value_type ub=1) {
    	randomize(lb, ub);
	}

   void randomize(value_type lb=0, value_type ub=1)  {
	   static_assert(internal_t::ITERATOR == 0 || internal_t::ITERATOR == 1,
			   	   	   "randomize not available to non-continuous tensors");

	   using impl = random::implementation<typename BC::allocator_traits<allocator_t>::system_tag>;
	   impl::randomize(this->as_derived().internal(), lb, ub);
   }
}; //end_of class 'Tensor_Functions'


}  //end_of namespace 'module'

#ifdef BC_CPP17 //------------------------------------------------------------------------------------------

template<class internal_t>
static auto sum(const Tensor_Base<internal_t>& tensor) {
	using p_value_type = typename internal_t::value_type;
	using sum_value_type = std::conditional_t<std::is_same<p_value_type, bool>::value, BC::size_t, p_value_type>;

	return BC::accumulate(tensor.cbegin(), tensor.cend(), sum_value_type(0));
}

template<class internal_t>
static bool prod(const Tensor_Base<internal_t>& tensor) {
	using value_type = typename internal_t::value_type;
	return BC::accumulate(tensor.cbegin(), tensor.cend(), value_type(1), BC::et::oper::mul());
}

template<class internal_t>
static bool all(const Tensor_Base<internal_t>& tensor) {
	return tensor.size() == sum(logical(tensor));
}

template<class internal_t>
static bool any(const Tensor_Base<internal_t>& tensor) {
	return sum(logical(tensor)) != 0;
}


template<class internal_t>
static auto max(const Tensor_Base<internal_t>& tensor) {
	static_assert(BC::is_array<internal_t>(), "'max' is only available to Array types, max on 'Expressions' is prohibited");
	auto max_index = BC::alg::max_element(tensor.cbegin(), tensor.cend());
	return tensor(max_index);
}

template<class internal_t>
static auto min(const Tensor_Base<internal_t>& tensor) {
	static_assert(BC::is_array<internal_t>(), "'min' is only available to Array types, min on 'Expressions' is prohibited");
	auto min_index = BC::alg::min_element(tensor.cbegin(), tensor.cend());
	return tensor(min_index);
}

#endif //ifdef BC_CPP17 //------------------------------------------------------------------------------------------


} //end_of namespace 'BC'

#endif /* TENSOR_FUNCTIONS_H_ */
