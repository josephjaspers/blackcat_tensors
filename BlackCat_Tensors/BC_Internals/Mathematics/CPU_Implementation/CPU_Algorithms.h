/*
 * CPU_Algorithms.h
 *
 *  Created on: Nov 7, 2018
 *      Author: joseph
 */

#ifndef CPU_ALGORITHMS_H_
#define CPU_ALGORITHMS_H_
#include <algorithm>

#define BC_CPU_ALGORITHM_DEF(function) \
template<class iter_begin, class iter_end, class functor>\
static auto function (iter_begin begin, iter_end end, functor func)\
{					\
	return std:: function (begin, end, func);				\
}
#define BC_CPU_ALGORITHM2_DEF(function) \
template<class iter_begin, class iter_end, class arg, class functor>\
static auto function (iter_begin begin, iter_end end, arg p_arg, functor func)\
{					\
	return std:: function (begin, end, p_arg, func);				\
}



namespace BC {
template<class core_lib>
struct CPU_Algorithms {

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
	BC_CPU_ALGORITHM_DEF(all_of)
	BC_CPU_ALGORITHM_DEF(any_of)
	BC_CPU_ALGORITHM_DEF(none_of)
	BC_CPU_ALGORITHM_DEF(for_each)
	BC_CPU_ALGORITHM_DEF(find)
	BC_CPU_ALGORITHM2_DEF(find_if)
	BC_CPU_ALGORITHM2_DEF(find_if_not)


	BC_CPU_ALGORITHM_DEF(find_end)
	BC_CPU_ALGORITHM_DEF(find_first_of)
	BC_CPU_ALGORITHM_DEF(adjacent_find)
	BC_CPU_ALGORITHM_DEF(count)
	BC_CPU_ALGORITHM2_DEF(count_if)
	//count if
	BC_CPU_ALGORITHM_DEF(mismatch)
	BC_CPU_ALGORITHM_DEF(equal)
	BC_CPU_ALGORITHM_DEF(is_permutation)
	BC_CPU_ALGORITHM_DEF(search)
	BC_CPU_ALGORITHM_DEF(search_n)


	//modifying-----------------------------------------

	BC_CPU_ALGORITHM_DEF(copy)
	BC_CPU_ALGORITHM_DEF(copy_n)
	BC_CPU_ALGORITHM2_DEF(adjacent_find)
	BC_CPU_ALGORITHM_DEF(copy_if)
	BC_CPU_ALGORITHM_DEF(copy_backward)
	BC_CPU_ALGORITHM_DEF(move)
	BC_CPU_ALGORITHM_DEF(move_backward)
	BC_CPU_ALGORITHM_DEF(swap)
	BC_CPU_ALGORITHM_DEF(swap_ranges)
	BC_CPU_ALGORITHM_DEF(iter_swap)
	BC_CPU_ALGORITHM_DEF(transform)
	BC_CPU_ALGORITHM_DEF(replace)
	BC_CPU_ALGORITHM_DEF(replace_if)
	BC_CPU_ALGORITHM_DEF(replace_copy)
	BC_CPU_ALGORITHM_DEF(replace_copy_if)
	BC_CPU_ALGORITHM_DEF(fill)

	BC_CPU_ALGORITHM_DEF(fill_n)
	BC_CPU_ALGORITHM_DEF(generate)
	BC_CPU_ALGORITHM_DEF(generate_n)

//
//	BC_CPU_ALGORITHM_DEF(remove)
//	BC_CPU_ALGORITHM_DEF(remove_if)

//	BC_CPU_ALGORITHM_DEF(remove_copy)
//	BC_CPU_ALGORITHM_DEF(remove_copy_if)
//	BC_CPU_ALGORITHM_DEF(unique)
//	BC_CPU_ALGORITHM_DEF(unique_copy)

};

}



#endif /* CPU_ALGORITHMS_H_ */
