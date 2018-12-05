/*
 * CPU_Algorithms.h
 *
 *  Created on: Nov 7, 2018
 *      Author: joseph
 */

#ifndef BC_ALGORITHMS_HOST_H_
#define BC_ALGORITHMS_HOST_H_
#include <numeric>
#include <algorithm>
#include "Common.h"
#if  defined(BC_CPP17_EXECUTION) && defined(BC_CPP_EXECUTION_POLICIES)
    #define BC_CPU_ALGORITHM_FORWARDER_DEF(function)\
    \
        template<class... args>\
        static auto function (args... parameters){\
            return std:: function (BC_CPP17_EXECUTION, parameters...);\
        }
#else
    #define BC_CPU_ALGORITHM_FORWARDER_DEF(function)\
    \
        template<class... args>\
        static auto function (args... parameters){\
            return std:: function (parameters...);\
        }
#endif


#define BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(function) \
template<class... args>  \
static auto function (args... parameters){ \
    static_assert(sizeof...(args) == 100, "BC::Host::Algorithms DOES NOT DEFINE: " #function );\
}


namespace BC {
namespace algorithms {
struct Host {

    //non-modifying sequences

    BC_CPU_ALGORITHM_FORWARDER_DEF(all_of)
    BC_CPU_ALGORITHM_FORWARDER_DEF(any_of)
    BC_CPU_ALGORITHM_FORWARDER_DEF(none_of)
    BC_CPU_ALGORITHM_FORWARDER_DEF(for_each)

    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(for_each_n) )

    BC_CPU_ALGORITHM_FORWARDER_DEF(count)
    BC_CPU_ALGORITHM_FORWARDER_DEF(count_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(find)
    BC_CPU_ALGORITHM_FORWARDER_DEF(find_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(find_if_not)
    BC_CPU_ALGORITHM_FORWARDER_DEF(find_end)
    BC_CPU_ALGORITHM_FORWARDER_DEF(find_first_of)
    BC_CPU_ALGORITHM_FORWARDER_DEF(adjacent_find)
    BC_CPU_ALGORITHM_FORWARDER_DEF(search)
    BC_CPU_ALGORITHM_FORWARDER_DEF(search_n)

    //modifying sequences
    BC_CPU_ALGORITHM_FORWARDER_DEF(copy)
    BC_CPU_ALGORITHM_FORWARDER_DEF(copy_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(copy_n)
    BC_CPU_ALGORITHM_FORWARDER_DEF(copy_backward)
    BC_CPU_ALGORITHM_FORWARDER_DEF(move)
    BC_CPU_ALGORITHM_FORWARDER_DEF(move_backward)
    BC_CPU_ALGORITHM_FORWARDER_DEF(fill)
    BC_CPU_ALGORITHM_FORWARDER_DEF(fill_n)
    BC_CPU_ALGORITHM_FORWARDER_DEF(transform)
    BC_CPU_ALGORITHM_FORWARDER_DEF(generate)
    BC_CPU_ALGORITHM_FORWARDER_DEF(generate_n)
    //    BC_CPU_ALGORITHM_FORWARDER_DEF(remove)
    //    BC_CPU_ALGORITHM_FORWARDER_DEF(remove_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(replace)
    BC_CPU_ALGORITHM_FORWARDER_DEF(replace_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(replace_copy)
    BC_CPU_ALGORITHM_FORWARDER_DEF(replace_copy_if)
    BC_CPU_ALGORITHM_FORWARDER_DEF(swap)
    BC_CPU_ALGORITHM_FORWARDER_DEF(swap_ranges)
    BC_CPU_ALGORITHM_FORWARDER_DEF(iter_swap)
    BC_CPU_ALGORITHM_FORWARDER_DEF(reverse)
    BC_CPU_ALGORITHM_FORWARDER_DEF(reverse_copy)
    BC_CPU_ALGORITHM_FORWARDER_DEF(rotate)
    BC_CPU_ALGORITHM_FORWARDER_DEF(rotate_copy)

    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(shift_left))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(shift_right))
    BC_CPU_ALGORITHM_FORWARDER_DEF(random_shuffle)
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(sample))
//  BC_will not define removing
//    BC_CPU_ALGORITHM_FORWARDER_DEF(unique)
//    BC_CPU_ALGORITHM_FORWARDER_DEF(unique_copy)

    //partition N/A-------------------
    //do not define any part of partitioning

    //Sorting
    BC_CPU_ALGORITHM_FORWARDER_DEF(is_sorted)
    BC_CPU_ALGORITHM_FORWARDER_DEF(is_sorted_until)
    BC_CPU_ALGORITHM_FORWARDER_DEF(sort)
    BC_CPU_ALGORITHM_FORWARDER_DEF(partial_sort)
    BC_CPU_ALGORITHM_FORWARDER_DEF(partial_sort_copy)
    BC_CPU_ALGORITHM_FORWARDER_DEF(stable_sort)
    BC_CPU_ALGORITHM_FORWARDER_DEF(nth_element)
    //searching
    BC_CPU_ALGORITHM_FORWARDER_DEF(lower_bound)
    BC_CPU_ALGORITHM_FORWARDER_DEF(upper_bound)
    BC_CPU_ALGORITHM_FORWARDER_DEF(binary_search)
    BC_CPU_ALGORITHM_FORWARDER_DEF(equal_range)
    //other
    //merge
    //inplace_merge
    BC_CPU_ALGORITHM_FORWARDER_DEF(max)
    BC_CPU_ALGORITHM_FORWARDER_DEF(max_element)
    BC_CPU_ALGORITHM_FORWARDER_DEF(min)
    BC_CPU_ALGORITHM_FORWARDER_DEF(min_element)
    BC_CPU_ALGORITHM_FORWARDER_DEF(minmax)
    BC_CPU_ALGORITHM_FORWARDER_DEF(minmax_element)
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(clamp))
    BC_CPU_ALGORITHM_FORWARDER_DEF(equal)
    BC_CPU_ALGORITHM_FORWARDER_DEF(lexicographical_compare)

    //numeric--------------------------
    BC_DEF_IF_CPP17(BC_CPU_ALGORITHM_FORWARDER_DEF(iota))
    BC_DEF_IF_CPP17(BC_CPU_ALGORITHM_FORWARDER_DEF(accumulate))
    BC_DEF_IF_CPP17(BC_CPU_ALGORITHM_FORWARDER_DEF(inner_product))
    BC_DEF_IF_CPP17(BC_CPU_ALGORITHM_FORWARDER_DEF(adjacent_difference))
    BC_DEF_IF_CPP17(BC_CPU_ALGORITHM_FORWARDER_DEF(partial_sum))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(reduce))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(exclusive_scan))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(inclusive_scan))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(transform_reduce))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(transform_exclusive_scan))
    BC_DEF_IF_CPP17(BC_ALGORITHM_HOST_NDEF_FORWARDER_DEF(transform_inclusive_scan))
    BC_CPU_ALGORITHM_FORWARDER_DEF(qsort)
    BC_CPU_ALGORITHM_FORWARDER_DEF(bsearch)


};
}
}



#endif /* CPU_ALGORITHMS_H_ */
