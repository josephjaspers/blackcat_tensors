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
struct Flatten : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using greedy_evaluate_delta = std::true_type;

	using input_tensor_dimension = InputTensorDimension;
	using output_tensor_dimension = BC::traits::Integer<1>;

	using requires_inputs = std::false_type;

private:

	BC::Shape<input_tensor_dimension::value> m_input_shape;
	BC::Shape<input_tensor_dimension::value+1> m_batched_input_shape;
	BC::Shape<1> m_output_shape;
	BC::Shape<2> m_batched_output_shape;

public:

	Flatten(BC::Shape<input_tensor_dimension::value> input_shape):
		Layer_Base(__func__, input_shape.size(), input_shape.size()),
		m_input_shape(input_shape),
		m_output_shape(input_shape.size()),
		m_batched_output_shape(input_shape.size(), 1)
	{
		set_batch_size(1);
	}

	template<class Matrix>
	auto forward_propagation(const Matrix& x) {
		return BC::reshape(x, m_batched_output_shape);
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		return BC::reshape(dy, m_batched_input_shape);
	}


	void set_batch_size(BC::size_t batch_sz) {
		Layer_Base::set_batch_size(batch_sz);

		BC::utility::array<input_tensor_dimension::value+1, BC::size_t> batched_shape;
		for (int i = 0; i < input_tensor_dimension::value; ++i) {
			batched_shape[i] = m_input_shape.dimension(i);
		}
		batched_shape[input_tensor_dimension::value] = batch_sz;
		m_batched_input_shape = BC::Shape<input_tensor_dimension::value+1>(batched_shape);
		m_batched_output_shape =  BC::Shape<2>(this->input_size(), batch_sz);
	}

	auto get_input_shape() const { return m_input_shape; }
	auto get_output_shape() const { return m_output_shape; }
	auto get_batched_input_shape() const { return m_batched_input_shape; }
	auto get_batched_output_shape() const { return m_batched_output_shape; }

};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag, int X>
auto flatten(SystemTag system_tag, Shape<X> shape) {
	return Flatten<SystemTag, ValueType, BC::traits::Integer<X>>(shape);
}
#endif

template<class SystemTag, int X>
auto flatten(SystemTag system_tag, BC::Shape<X> shape) {
	return Flatten<SystemTag, typename SystemTag::default_floating_point_type, BC::traits::Integer<X>>(shape);
}

template<int X>
auto flatten(BC::Shape<X> shape) {
	return Flatten<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type, BC::traits::Integer<X>>(shape);
}


}
}

#endif /* FEEDFORWARD_CU_ */
