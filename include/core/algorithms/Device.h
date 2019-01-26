/*
 * GPU_Algorithms.h
 *
 *  Created on: Nov 7, 2018
 *      Author: joseph
 */

#ifdef __CUDACC__
#ifndef BC_ALGORITHMS_DEVICE_H_
#define BC_ALGORITHMS_DEVICE_H_

#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdlib>

#include "Common.h"

//forwards methods to thrust implementation
#define BC_GPU_ALGORITHM_FORWARDER_DEF(function) \
template<class... args>\
static auto function (args... parameters){\
    return thrust:: function (thrust::device, parameters...);\
}

//Gives an appropriate error (for algorithms not defined in thrust)
#define BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(function) \
template<class... args>  \
static auto function (args... parameters){ \
    static_assert(sizeof...(args) == 100, "(CUDA) THRUST DOES NOT DEFINE: " #function );\
}

//Thrust is wonky with stl-style iterators and sorting.
//For now just send it the pointers.
//Will need to fix once multidimensional iterators are added.
//This is simply a tmp-fix hence -hack
#define BC_GPU_ALGORITHM_FORWARDER_DEF_2PTR_HACK(function) \
template<class iter_begin, class iter_end>\
static auto function (iter_begin begin, iter_end end){\
    return thrust:: function (thrust::device, &begin[begin.index], &end[end.index]);\
}



namespace BC {
namespace algorithms {
struct Device {

    //non-modifying sequences
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(all_of)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(any_of)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(none_of)
    BC_GPU_ALGORITHM_FORWARDER_DEF(for_each)

//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(for_each_n) )

    BC_GPU_ALGORITHM_FORWARDER_DEF(count)
    BC_GPU_ALGORITHM_FORWARDER_DEF(count_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(find)
    BC_GPU_ALGORITHM_FORWARDER_DEF(find_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(find_if_not)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(find_end)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(find_first_of)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(adjacent_find)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(search)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(search_n)

    //modifying sequences
    BC_GPU_ALGORITHM_FORWARDER_DEF(copy)
    BC_GPU_ALGORITHM_FORWARDER_DEF(copy_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(copy_n)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(copy_backward)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(move)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(move_backward)
    BC_GPU_ALGORITHM_FORWARDER_DEF(fill)
    BC_GPU_ALGORITHM_FORWARDER_DEF(fill_n)
    BC_GPU_ALGORITHM_FORWARDER_DEF(transform)
    BC_GPU_ALGORITHM_FORWARDER_DEF(generate)
    BC_GPU_ALGORITHM_FORWARDER_DEF(generate_n)
    //    BC_GPU_ALGORITHM_FORWARDER_DEF(remove)
    //    BC_GPU_ALGORITHM_FORWARDER_DEF(remove_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(replace)
    BC_GPU_ALGORITHM_FORWARDER_DEF(replace_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(replace_copy)
    BC_GPU_ALGORITHM_FORWARDER_DEF(replace_copy_if)
    BC_GPU_ALGORITHM_FORWARDER_DEF(swap)
    BC_GPU_ALGORITHM_FORWARDER_DEF(swap_ranges)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(iter_swap)
    BC_GPU_ALGORITHM_FORWARDER_DEF(reverse)
    BC_GPU_ALGORITHM_FORWARDER_DEF(reverse_copy)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(rotate)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(rotate_copy)


//
//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(shift_left))
//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(shift_right))
//    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(random_shuffle)
//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(sample))
//  BC_will not define removing
//    BC_GPU_ALGORITHM_FORWARDER_DEF(unique)
//    BC_GPU_ALGORITHM_FORWARDER_DEF(unique_copy)

    //partition N/A-------------------
    //do not define any part of partitioning

    //Sorting
    BC_GPU_ALGORITHM_FORWARDER_DEF_2PTR_HACK(is_sorted)
    BC_GPU_ALGORITHM_FORWARDER_DEF_2PTR_HACK(is_sorted_until)
    BC_GPU_ALGORITHM_FORWARDER_DEF_2PTR_HACK(sort)

    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(partial_sort)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(partial_sort_copy)
    BC_GPU_ALGORITHM_FORWARDER_DEF(stable_sort)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(nth_element)
    //searching
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(lower_bound)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(upper_bound)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(binary_search)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(equal_range)
    //other
    //merge
    //inplace_merge
//    BC_GPU_ALGORITHM_FORWARDER_DEF(max)
    BC_GPU_ALGORITHM_FORWARDER_DEF(max_element)
//    BC_GPU_ALGORITHM_FORWARDER_DEF(min)
    BC_GPU_ALGORITHM_FORWARDER_DEF(min_element)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(minmax)
    BC_GPU_ALGORITHM_FORWARDER_DEF(minmax_element)
//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(clamp))
    BC_GPU_ALGORITHM_FORWARDER_DEF(equal)
    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(lexicographical_compare)

    //numeric--------------------------
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(iota))
//    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(accumulate))

    template<class... args>\
    static auto accumulate (args... parameters){\
        return thrust:: reduce (thrust::device, parameters...);\
    }


    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(inner_product))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(adjacent_difference))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(partial_sum))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(reduce))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(exclusive_scan))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(inclusive_scan))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_FORWARDER_DEF(transform_reduce))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(transform_exclusive_scan))
    BC_DEF_IF_CPP17(BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(transform_inclusive_scan))
//    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(qsort)
//    BC_GPU_ALGORITHM_THRUST_NDEF_FORWARDER_DEF(bsearch)


};
}
}



#endif /* GPU_ALGORITHMS_H_ */
#endif
