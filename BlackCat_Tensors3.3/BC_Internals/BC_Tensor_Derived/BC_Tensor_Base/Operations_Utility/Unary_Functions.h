/*
 * Base_Functions.h
 *
 *  Created on: May 15, 2018
 *      Author: joseph
 */

#ifndef BASE_FUNCTIONS_H_
#define BASE_FUNCTIONS_H_

//This should be included at the bottom of Tensor_Base.h
//Reshape and Chunk are curried functions
//This file defines a set of unary_functions
namespace BC {

template<class> class Tensor_Base;

//Returns the tensor reshaped, does not modify the original tensor-dimensions
template<class T> __BC_host_inline__
auto reshape(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) { return tensor.self_reshape(integers...); };
}
template<class T> __BC_host_inline__
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) { return tensor.self_reshape(integers...); };
}
//Returns vc da 'chunk' of the tensor, the first set of parameters being the index, the second the shpe
template<class T> __BC_host_inline__
auto chunk(const Tensor_Base<T>& tensor) {
	return [&](auto... location_indices) {
		return [&] (auto... chunk_dimension) {
			return tensor.self_chunk(location_indices...)(chunk_dimension...); };
		};
}
template<class T> __BC_host_inline__
auto chunk(Tensor_Base<T>& tensor) {
	return [&](auto... location_indices) {
		return [&] (auto... chunk_dimension) {
			return tensor.self_chunk(location_indices...)(chunk_dimension...); };
		};
}
//zero and one are the equivalent but delayed evaluation version,
//modifies the calling the tensor, but if and only if evaluated
template<class deriv>
static auto zero(Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::zero());
}
template<class deriv>
static auto one(Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::one());
}

//converts all NaN/Inf values to 0
template<class deriv>
static auto fix(Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::fix());
}
template<class deriv>
static auto abs(const Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::abs());
}
template<class deriv>
static auto negation(const Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::negation());
}

//if 0 return 0 else return 1
template<class deriv>
static auto logical(const Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::logical());
}

}


#endif /* BASE_FUNCTIONS_H_ */
