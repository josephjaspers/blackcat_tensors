/*
 * Flatten.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FLATTEN_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FLATTEN_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class InputTensorDimension=BC::traits::Integer<3>>
struct Flatten:
		public Layer_Base<
				Flatten<SystemTag, ValueType, InputTensorDimension>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using parent_type = Layer_Base<
			Flatten<SystemTag, ValueType, InputTensorDimension>>;

	using greedy_evaluate_delta = std::true_type;
	using input_tensor_dimension = InputTensorDimension;
	using output_tensor_dimension = BC::traits::Integer<1>;
	using requires_inputs = std::false_type;

private:

	Dim<input_tensor_dimension::value> m_input_shape;

public:

	Flatten(BC::Dim<input_tensor_dimension::value> input_shape):
		parent_type(__func__, input_shape.size(), input_shape.size()),
		m_input_shape(input_shape) {}

	template<class Matrix>
	auto forward_propagation(const Matrix& x) {
		return BC::reshape(x, this->get_batched_output_shape());
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		return BC::reshape(dy, this->get_batched_input_shape());
	}

	auto get_input_shape() const { return m_input_shape; }
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag, int X>
auto flatten(SystemTag system_tag, Dim<X> shape) {
	return Flatten<SystemTag, ValueType, BC::traits::Integer<X>>(shape);
}
#endif

template<class SystemTag, int X>
auto flatten(SystemTag system_tag, Dim<X> shape) {
	return Flatten<
			SystemTag,
			typename SystemTag::default_floating_point_type,
			BC::traits::Integer<X>>(shape);
}

template<int X>
auto flatten(Dim<X> shape) {
	return Flatten<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type,
			BC::traits::Integer<X>>(shape);
}


}
}

#endif /* FEEDFORWARD_CU_ */
