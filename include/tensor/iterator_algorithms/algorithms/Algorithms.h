/*
 * algorithms.h
 *
 *  Created on: Nov 25, 2018
 *      Author: joseph
 */

#ifndef BC_AGORITHMS_ALGORITHMS_H_
#define BC_AGORITHMS_ALGORITHMS_H_


BC_DEFAULT_MODULE_BODY(algorithms, Algorithm)

#include "Device.h"
#include "Host.h"


namespace BC {

#define BC_TENSOR_ALGORITHM_DEF(function)\
\
    template<class iter_begin_, class iter_end_, class... args>\
    static auto function (iter_begin_ begin_, iter_end_ end_, args... params) {\
        using tensor_t = typename iter_end_::tensor_t;\
        using system_tag  = typename tensor_t::system_tag;\
        return algorithms::Algorithm<system_tag>:: function (begin_, end_, params...);\
    }\

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
}
#endif /* ALGORITHMS_H_ */
