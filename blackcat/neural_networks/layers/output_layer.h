/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "layer_base.h"

namespace bc {
namespace nn {

template<class SystemTag, class ValueType>
struct Output_Layer:
		Layer_Base<Output_Layer<SystemTag, ValueType>,
		Tensor_Descriptor<ValueType, SystemTag, Integer<1>>>
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using self_type = Output_Layer<SystemTag, ValueType>;
	using input_descriptor_t = Tensor_Descriptor<ValueType, SystemTag, Integer<1>>;
	using parent_type = Layer_Base<self_type, input_descriptor_t>;

	Output_Layer(int inputs):
		parent_type(__func__, {inputs}, {inputs}) {}

	template <class Tensor>
	const auto forward_propagation(const Tensor& x) {
		return x.shallow_copy();
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
