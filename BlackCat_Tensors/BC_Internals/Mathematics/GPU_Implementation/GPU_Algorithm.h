/*
 * GPU_Algorithms.h
 *
 *  Created on: Nov 7, 2018
 *      Author: joseph
 */

#ifndef GPU_ALGORITHMS_H_
#define GPU_ALGORITHMS_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#define BC_GPU_ALGORITHM_DEF(function) \
template<class iter_begin, class iter_end, class functor>\
static auto function (iter_begin begin, iter_end end, functor func)\
{					\
	return thrust:: function (begin, end, func);				\
}
#define BC_GPU_ALGORITHM2_DEF(function) \
template<class iter_begin, class iter_end, class arg, class functor>\
static auto function (iter_begin begin, iter_end end, arg p_arg, functor func)\
{					\
	return thrust:: function (begin, end, p_arg, func);				\
}

namespace BC {
template<class core_lib>
struct GPU_Algorithm {
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

	//THe commented out ones are not supported by thrust
//	BC_GPU_ALGORITHM_DEF(all_of)
//	BC_GPU_ALGORITHM_DEF(any_of)
//	BC_GPU_ALGORITHM_DEF(none_of)
	BC_GPU_ALGORITHM_DEF(for_each)
	BC_GPU_ALGORITHM_DEF(find)
	BC_GPU_ALGORITHM2_DEF(find_if)
	BC_GPU_ALGORITHM2_DEF(find_if_not)


//	BC_GPU_ALGORITHM_DEF(find_end)
//	BC_GPU_ALGORITHM_DEF(find_first_of)
//	BC_GPU_ALGORITHM_DEF(adjacent_find)
	BC_GPU_ALGORITHM_DEF(count)
	BC_GPU_ALGORITHM2_DEF(count_if)
	//count if
	BC_GPU_ALGORITHM_DEF(mismatch)
	BC_GPU_ALGORITHM_DEF(equal)
//	BC_GPU_ALGORITHM_DEF(is_permutation)
//	BC_GPU_ALGORITHM_DEF(search)
//	BC_GPU_ALGORITHM_DEF(search_n)


	//modifying-----------------------------------------

	BC_GPU_ALGORITHM_DEF(copy)
	BC_GPU_ALGORITHM_DEF(copy_n)
//	BC_GPU_ALGORITHM2_DEF(adjacent_find)
	BC_GPU_ALGORITHM_DEF(copy_if)
//	BC_GPU_ALGORITHM_DEF(copy_backward)
//	BC_GPU_ALGORITHM_DEF(move)
//	BC_GPU_ALGORITHM_DEF(move_backward)
	BC_GPU_ALGORITHM_DEF(swap)
	BC_GPU_ALGORITHM_DEF(swap_ranges)
//	BC_GPU_ALGORITHM_DEF(iter_swap)
	BC_GPU_ALGORITHM_DEF(transform)
	BC_GPU_ALGORITHM_DEF(replace)
	BC_GPU_ALGORITHM_DEF(replace_if)
	BC_GPU_ALGORITHM_DEF(replace_copy)
	BC_GPU_ALGORITHM_DEF(replace_copy_if)
	BC_GPU_ALGORITHM_DEF(fill)

	BC_GPU_ALGORITHM_DEF(fill_n)
	BC_GPU_ALGORITHM_DEF(generate)
	BC_GPU_ALGORITHM_DEF(generate_n)

};
}



#endif /* GPU_ALGORITHMS_H_ */
