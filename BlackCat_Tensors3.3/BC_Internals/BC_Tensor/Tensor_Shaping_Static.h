/*
 * Shaping_Functions.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef SHAPING_FUNCTIONS_H_
#define SHAPING_FUNCTIONS_H_

namespace BC{

template<class> class Tensor_Base;

template<class T> __BC_host_inline__
const auto reshape(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		using mathlib_type = _mathlib<T>;

	using internal = decltype(std::declval<T>()._reshape(integers...));
	return Tensor_Base<internal>(tensor._reshape(integers...));
	};
}
template<class T> __BC_host_inline__
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		using mathlib_type = _mathlib<T>;

	using internal = decltype(std::declval<T>()._reshape(integers...));
	return Tensor_Base<internal>(tensor._reshape(integers...));
	};
}

template<class T> __BC_host_inline__
auto chunk(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		int location = tensor.dims_to_index_reverse(integers...);
		return typename Tensor_Base<T>::CHUNK(tensor, location);
	};
}
template<class T> __BC_host_inline__
const auto chunk(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		int location = tensor.dims_to_index_reverse(integers...);
		return typename Tensor_Base<T>::CHUNK(tensor, location);
	};
}

}



#endif /* SHAPING_FUNCTIONS_H_ */
