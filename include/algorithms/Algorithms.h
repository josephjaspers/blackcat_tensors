/*
 * algorithms.h
 *
 *  Created on: Nov 25, 2018
 *      Author: joseph
 */

#ifndef BC_ALGORITHMS_ALGORITHMS_H_
#define BC_ALGORITHMS_ALGORITHMS_H_

#include "Common.h"
#include <numeric>
#include <algorithm>

#ifdef __CUDACC__
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#endif

BC_DEFAULT_MODULE_BODY(algorithms, Algorithm)

namespace BC {
namespace algorithms {


#define BC_ALGORITHM_DEF(function)\
BC_IF_CUDA(\
template<class Begin, class End, class... Args>\
static auto function (BC::streams::Stream<BC::device_tag> stream, Begin begin, End end, Args... args) {\
	return thrust::function(thrust::cuda::par.on(stream), begin, end, args...);\
}\
template<class Begin, class End, class... Args>\
static auto function (cudaStream_t stream, Begin begin, End end, Args... args) {\
	return thrust::function(thrust::cuda::par.on(stream), begin, end, args...);\
})\
template<class Begin, class End, class... Args>\
static auto function (BC::streams::Stream<BC::host_tag> stream, Begin begin, End end, Args... args) {   \
	return stream.enqueue([&](){std::function(begin, end, args...); });\
}\
template<class Container, class... Args>\
static auto function (const Container& container, Args&&... args) {\
	return function(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);\
}\
template<class Container, class... Args>\
static auto function (Container& container, Args&&... args) {\
	return function(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);\
}\

#define BC_REDUCE_ALGORITHM_DEF(function)\
BC_IF_CUDA(\
template<class Begin, class End, class... Args>\
static auto function (const BC::streams::Stream<BC::device_tag>& stream, Begin begin, End end, Args... args) {\
	return thrust::function(thrust::cuda::par.on(stream), begin, end, args...);\
}\
template<class Begin, class End, class... Args>\
static auto function (cudaStream_t stream, Begin begin, End end, Args... args) {\
	return thrust::function(thrust::cuda::par.on(stream), begin, end, args...);\
})\
template<class Begin, class End, class... Args>\
static auto function (BC::streams::Stream<BC::host_tag> stream, Begin begin, End end, Args... args) {   \
	double value = 1.0;\
	stream.enqueue([&](){ value = std::function(begin, end, args...); });\
	stream.sync();\
	return value;\
}\
template<class Container, class... Args>\
static auto function (const Container& container, Args&&... args) {\
	return function(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);\
}\
template<class Container, class... Args>\
static auto function (Container& container, Args&&... args) {\
	return function(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);\
}\

//---------------------------non-modifying sequences---------------------------//
//BC_ALGORITHM_DEF(all_of)
//BC_ALGORITHM_DEF(any_of)
//BC_ALGORITHM_DEF(none_of)
BC_ALGORITHM_DEF(for_each)
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(for_each_n) )
BC_ALGORITHM_DEF(count)
BC_ALGORITHM_DEF(count_if)
BC_ALGORITHM_DEF(find)
BC_ALGORITHM_DEF(find_if)
BC_ALGORITHM_DEF(find_if_not)
//BC_ALGORITHM_DEF(find_end)
//BC_ALGORITHM_DEF(find_first_of)
//BC_ALGORITHM_DEF(adjacent_find)
//BC_ALGORITHM_DEF(search)
//BC_ALGORITHM_DEF(search_n)
//modifying sequences
BC_ALGORITHM_DEF(copy)
BC_ALGORITHM_DEF(copy_if)
BC_ALGORITHM_DEF(copy_n)
//BC_ALGORITHM_DEF(copy_backward)
//BC_ALGORITHM_DEF(move)
//BC_ALGORITHM_DEF(move_backward)
BC_ALGORITHM_DEF(fill)
BC_ALGORITHM_DEF(fill_n)
BC_ALGORITHM_DEF(transform)
BC_ALGORITHM_DEF(generate)
BC_ALGORITHM_DEF(generate_n)
BC_ALGORITHM_DEF(replace)
BC_ALGORITHM_DEF(replace_if)
BC_ALGORITHM_DEF(replace_copy)
BC_ALGORITHM_DEF(replace_copy_if)
BC_ALGORITHM_DEF(swap)
BC_ALGORITHM_DEF(swap_ranges)
//BC_ALGORITHM_DEF(iter_swap)
BC_ALGORITHM_DEF(reverse)
BC_ALGORITHM_DEF(reverse_copy)
//BC_ALGORITHM_DEF(rotate)
//BC_ALGORITHM_DEF(rotate_copy)

//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(shift_left))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(shift_right))
//BC_ALGORITHM_DEF(random_shuffle)
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(sample))
//--------------------------- sorting ---------------------------//
BC_ALGORITHM_DEF(is_sorted)
BC_ALGORITHM_DEF(is_sorted_until)
BC_ALGORITHM_DEF(sort)
//BC_ALGORITHM_DEF(partial_sort)
//BC_ALGORITHM_DEF(partial_sort_copy)
BC_ALGORITHM_DEF(stable_sort)
//BC_ALGORITHM_DEF(nth_element)
//BC_ALGORITHM_DEF(lower_bound)
//BC_ALGORITHM_DEF(upper_bound)
//BC_ALGORITHM_DEF(binary_search)
//BC_ALGORITHM_DEF(equal_range)

//--------------------------- min/max ---------------------------//
BC_REDUCE_ALGORITHM_DEF(max)
BC_REDUCE_ALGORITHM_DEF(max_element)
BC_REDUCE_ALGORITHM_DEF(min)
BC_REDUCE_ALGORITHM_DEF(min_element)
//BC_ALGORITHM_DEF(minmax)
BC_REDUCE_ALGORITHM_DEF(minmax_element)
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(clamp))
BC_ALGORITHM_DEF(equal)
//BC_ALGORITHM_DEF(lexicographical_compare)

//--------------------------- numeric (mostly undefined) ---------------------------//
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(iota))

BC_IF_CUDA(
	template<class Begin, class End, class... Args>
	static auto accumulate (BC::streams::Stream<BC::device_tag> stream, Begin begin, End end, Args... args) {
		return thrust::reduce(thrust::cuda::par.on(stream), begin, end, args...);
	}
	template<class Begin, class End, class... Args>
	static auto accumulate (cudaStream_t stream, Begin begin, End end, Args... args) {
		return thrust::reduce(thrust::cuda::par.on(stream), begin, end, args...);
	}
)

template<class Begin, class End, class... Args>
static auto accumulate (BC::streams::Stream<BC::host_tag> stream, Begin begin, End end, Args... args) {
	double value = 1.0;
	stream.enqueue([&](){ value = std::accumulate(begin, end, args...); });
	stream.sync();
	return value;
}

template<class Container, class... Args>
static auto accumulate (const Container& container, Args&&... args) {
	return accumulate(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);
}

template<class Container, class... Args>
static auto accumulate (Container& container, Args&&... args) {
	return accumulate(BC::streams::select_on_get_stream(container), container.begin(), container.end(), args...);
}


//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(inner_product))
BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(adjacent_difference))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(partial_sum))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(reduce))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(exclusive_scan))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(inclusive_scan))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(transform_reduce))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(transform_exclusive_scan))
//BC_DEF_IF_CPP17(BC_ALGORITHM_DEF(transform_inclusive_scan))
//BC_ALGORITHM_DEF(qsort)
//BC_ALGORITHM_DEF(bsearch)

#undef BC_ALGORITHM_DEF
#undef BC_REDUCE_ALGORITHM_DEF

} //ns algorithms
} //bs BC
#endif /* ALGORITHMS_H_ */
