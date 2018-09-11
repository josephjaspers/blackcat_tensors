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
template<class internal_t>
auto make_tensor(internal_t internal) {
	return Tensor_Base<internal_t>(internal);
}

template<class T>
const auto reshape(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(tensor._reshape(integers...));
	};
}
template<class T>
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(tensor._reshape(integers...));
	};
}

template<class T> __BC_host_inline__
auto chunk(Tensor_Base<T>& tensor) {
	return [&](BC::array<T::DIMS(), int> point, auto... array_shape) {
		return make_tensor(tensor._chunk(point, make_array(array_shape...)));
	};
}
template<class T> __BC_host_inline__
const auto chunk(const Tensor_Base<T>& tensor) {
	return [&](BC::array<T::DIMS(), int> point, auto... array_shape) {
		return make_tensor(tensor._chunk(point, make_array(array_shape...)));
	};
}

}



#endif /* SHAPING_FUNCTIONS_H_ */
