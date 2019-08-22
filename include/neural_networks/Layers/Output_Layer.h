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
struct Output_Layer : Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;

	Output_Layer(int inputs):
		Layer_Base(__func__, inputs, inputs) {}

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
Output_Layer<SystemTag, ValueType> output_layer(SystemTag system_tag, int inputs) {
	return Output_Layer<SystemTag, ValueType>(inputs);
}
template<class SystemTag>
auto output_layer(SystemTag system_tag, int inputs) {
	return Output_Layer<SystemTag, typename SystemTag::default_floating_point_type>(inputs);
}
#endif

auto output_layer(int inputs) {
	return Output_Layer<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs);
}


}
}



#endif /* FEEDFORWARD_CU_ */
