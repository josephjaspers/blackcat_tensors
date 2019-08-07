/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
struct OutputLayer : Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;

	OutputLayer(int inputs):
		Layer_Base(inputs, inputs) {}

	template <class Tensor>
	const auto& forward_propagation(const Tensor& x) {
		return x;
	}

	template <class TensorX, class TensorY>
	auto back_propagation(const TensorX& x, const TensorY& y) {
		return x - y;
	}
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
OutputLayer<SystemTag, ValueType> outputlayer(SystemTag system_tag, int inputs) {
	return OutputLayer<SystemTag, ValueType>(inputs);
}
template<class SystemTag>
auto outputlayer(SystemTag system_tag, int inputs) {
	return OutputLayer<SystemTag, typename SystemTag::default_floating_point_type>(inputs);
}
#endif

auto outputlayer(int inputs) {
	return OutputLayer<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs);
}


}
}



#endif /* FEEDFORWARD_CU_ */
