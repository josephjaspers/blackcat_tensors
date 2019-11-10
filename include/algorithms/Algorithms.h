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
#include <thrust/reduce.h>
#endif

BC_DEFAULT_MODULE_BODY(algorithms, Algorithm)

namespace BC {
namespace algorithms {

#define BC_ALGORITHM_DEF(function)                                          \
                                                                            \
BC_IF_CUDA(                                                                 \
template<class Begin, class End, class... Args>                             \
static auto function(                                                       \
		BC::streams::Stream<BC::device_tag> stream,                         \
		Begin begin,                                                        \
		End end,                                                            \
		Args... args)                                                       \
{                                                                           \
	return thrust::function(                                                \
			thrust::cuda::par.on(stream), begin, end, args...);             \
})                                                                          \
                                                                            \
template<class Begin, class End, class... Args>                             \
static auto function (                                                      \
		BC::streams::Stream<BC::host_tag> stream,                           \
		Begin begin,                                                        \
		End end,                                                            \
		Args... args)                                                       \
{                                                                           \
	return stream.enqueue([&](){std::function(begin, end, args...); });     \
}

#define BC_REDUCE_ALGORITHM_DEF(function)                                 \
BC_IF_CUDA(                                                               \
template<class Begin, class End, class... Args>                           \
static auto function(                                                     \
		BC::streams::Stream<BC::device_tag> stream,                       \
		Begin begin,                                                      \
		End end,                                                          \
		Args... args)                                                     \
{                                                                         \
	return thrust::function(                                              \
			thrust::cuda::par.on(stream), begin, end, args...);           \
})                                                                        \
                                                                          \
template<class Begin, class End, class... Args>                           \
static auto function (                                                    \
		BC::streams::Stream<BC::host_tag> stream,                         \
		Begin begin,                                                      \
		End end,                                                          \
		Args... args)                                                     \
{                                                                         \
	double value = 1.0;                                                   \
	stream.enqueue([&](){ value = std::function(begin, end, args...); }); \
	stream.sync();                                                        \
	return value;                                                         \
}                                                                         \

//---------------------------non-modifying sequences---------------------------//
//BC_ALGORITHM_DEF(all_of)
//BC_ALGORITHM_DEF(any_of)
//BC_ALGORITHM_DEF(none_of)
BC_ALGORITHM_DEF(for_each)
BC_ALGORITHM_DEF(count)
BC_ALGORITHM_DEF(count_if)
BC_ALGORITHM_DEF(find)
BC_ALGORITHM_DEF(find_if)
BC_ALGORITHM_DEF(find_if_not)

//modifying sequences
BC_ALGORITHM_DEF(copy)
BC_ALGORITHM_DEF(copy_if)
BC_ALGORITHM_DEF(copy_n)
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
BC_ALGORITHM_DEF(reverse)
BC_ALGORITHM_DEF(reverse_copy)

//--------------------------- sorting ---------------------------//
BC_ALGORITHM_DEF(is_sorted)
BC_ALGORITHM_DEF(is_sorted_until)
BC_ALGORITHM_DEF(sort)
BC_ALGORITHM_DEF(stable_sort)

//--------------------------- min/max ---------------------------//
BC_REDUCE_ALGORITHM_DEF(max)
BC_REDUCE_ALGORITHM_DEF(max_element)
BC_REDUCE_ALGORITHM_DEF(min)
BC_REDUCE_ALGORITHM_DEF(min_element)
//BC_ALGORITHM_DEF(minmax)
BC_REDUCE_ALGORITHM_DEF(minmax_element)

#ifdef __CUDACC__

template<class Begin, class End, class... Args>
static auto accumulate (
		BC::streams::Stream<BC::device_tag> stream,
		Begin begin,
		End end,
		Args... args)
{
	return thrust::reduce(
			thrust::cuda::par.on(stream), begin, end, args...);
}

template<class Begin, class End, class... Args>
static auto accumulate (
		cudaStream_t stream,
		Begin begin,
		End end,
		Args... args)
{
	return thrust::reduce(
			thrust::cuda::par.on(stream), begin, end, args...);
}

#endif //#ifdef __CUDACC__

template<class Begin, class End, class... Args>
static auto accumulate (
		BC::streams::Stream<BC::host_tag> stream,
		Begin begin,
		End end,
		Args... args)
{
	double value = 1.0;
	stream.enqueue([&](){ value = std::accumulate(begin, end, args...); });
	stream.sync();
	return value;
}

template<class Container, class... Args>
static auto accumulate (const Container& container, Args&&... args)
{
	return accumulate(
			BC::streams::select_on_get_stream(container),
			container.cw_begin(),
			container.cw_end(),
			std::forward(args)...);
}

template<class Container, class... Args>
static auto accumulate (Container& container, Args&&... args)
{
	return accumulate(
			BC::streams::select_on_get_stream(container),
			container.cw_begin(),
			container.cw_end(),
			std::forward(args)...);
}

#undef BC_ALGORITHM_DEF
#undef BC_REDUCE_ALGORITHM_DEF

} //ns algorithms
} //bs BC

#endif /* ALGORITHMS_H_ */
