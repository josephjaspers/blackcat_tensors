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
	return [&](auto... integers) { return tensor.reshape_impl(integers...); };
}
template<class T> __BC_host_inline__
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) { return tensor.reshape_impl(integers...); };
}

template<class T> __BC_host_inline__
auto chunk(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return tensor.chunk_impl(integers...);
	};
}
template<class T> __BC_host_inline__
const auto chunk(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return tensor.chunk_impl(integers...);
	};
}

template<class T> __BC_host_inline__
const auto reshape(const Tensor_Base<T>&& tensor) {
	return [&](auto... integers) { return tensor.reshape_impl(integers...); };
}
template<class T> __BC_host_inline__
auto reshape(Tensor_Base<T>&& tensor) {
	return [&](auto... integers) { return tensor.reshape_impl(integers...); };
}
template<class T> __BC_host_inline__
auto chunk(Tensor_Base<T>&& tensor) {
	return [&](auto... integers) {
		return tensor.chunk_impl(integers...);
	};
}
template<class T> __BC_host_inline__
const auto chunk(const Tensor_Base<T>&& tensor) {
	return [&](auto... integers) {
		return tensor.chunk_impl(integers...);
	};
}

}



#endif /* SHAPING_FUNCTIONS_H_ */
