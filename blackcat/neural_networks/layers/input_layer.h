#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_INPUT_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_INPUT_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Dimension>
struct Input_Layer:
		Layer_Base<Dimension, ValueType, SystemTag>
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<Dimension, ValueType, SystemTag>;

	using self_type = Input_Layer<SystemTag, ValueType, Dimension>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;

	using greedy_evaluate_delta = std::true_type;

private:

	using typename parent_type::batched_output_tensor_type;
	using typename parent_type::batched_input_tensor_type;

	batched_input_tensor_type x;

public:

	Input_Layer(bc::Dim<Dimension::value> input_shape):
		parent_type(__func__)
	{
		this->m_input_shape = input_shape;
		this->m_output_shape = input_shape;
	}

	template<class... InputShapeInts>
	Input_Layer(InputShapeInts... dims):
		parent_type(__func__)
	{
		static_assert(sizeof...(InputShapeInts) == Dimension::value,
				"Input_Layer constructor 'Input_Layer(ints... shape_dims)` requires number of params equal the input_shape dimension");

		this->m_input_shape = bc::dim(dims...);
		this->m_output_shape = this->m_input_shape;
	}


	virtual ~Input_Layer()=default;

	void init() override
	{
		this->x = batched_input_tensor_type(this->batched_input_shape());
		this->y = batched_output_tensor_type(this->batched_output_shape());
	}

	virtual batched_output_tensor_type forward_propagation(
			const batched_input_tensor_type& x) override
	{
		return this->x = x;
	}

	virtual batched_input_tensor_type back_propagation(
			const batched_output_tensor_type& dy) override
	{
		return dy;
	}

	virtual void set_learning_rate_hook(double lr) override {}
	void update_weights() {}
	virtual void save(Layer_Loader& loader) const override {}
	virtual void load(Layer_Loader& loader) override {}
};

template<class Dtype, class SystemTag, int N>
auto input_layer(bc::traits::Type<Dtype>, SystemTag, bc::Dim<N> shape) {
	return Input_Layer<SystemTag, Dtype, bc::traits::Integer<N>>(shape);
}

}
}

#endif /* FEEDFORWARD_CU_ */
