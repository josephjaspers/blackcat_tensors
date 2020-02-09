#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_INPUT_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_INPUT_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Dimension=bc::traits::Integer<1>>
struct Input_Layer:
		Layer_Base<Dimension, ValueType, SystemTag>
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<Dimension, ValueType, SystemTag>;

	using self_type = Input_Layer<SystemTag, ValueType, Dimension>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using optimizer_type = Optimizer;

	using greedy_evaluate_delta = std::true_type;

private:

	using typename parent_type::batched_output_tensor_type;
	using typename parent_type::batched_input_tensor_type;

	using mat = bc::Matrix<value_type, allocator_type>;
	using vec = bc::Vector<value_type, allocator_type>;

	using mat_opt_t = typename Optimizer::template Optimizer<mat>;
	using vec_opt_t = typename Optimizer::template Optimizer<vec>;

	NonlinearityFunction g;

	mat w; //weights
	vec b; //biases

	mat w_gradients;
	vec b_gradients;

	mat_opt_t w_opt;
	vec_opt_t b_opt;

public:

	Input_Layer(bc::Dim<Dimension> input_shape):
		parent_type(__func__)
	{
		this->m_input_shape = input_shape;
		this->m_output_shape = input_shape;
	}

	template<class... InputShapeInts>
	Input_Layer(InputShapeInts... dims):
		parent_type(__func__)
	{
		static_assert(sizeof...(InputShapeInts) == Dimension, "Input_Layer constructor 'Input_Layer(ints... shape_dims)` requires number of params equal the input_shape dimension");

		this->m_input_shape = bc::dims(dims);
		this->m_output_shape = m_input_shape;
	}


	virtual ~Input_Layer()=default;

	void init() override
	{
		x = batched_input_tensor_type(this->batched_input_shape());
		y = batched_output_tensor_type(this->batched_output_shape());
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
	Input_Layer<SystemTag, Dtype, bc::traits::Integer<N>>(shape);
}

}
}

#endif /* FEEDFORWARD_CU_ */
