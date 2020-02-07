/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Optimizer=Stochastic_Gradient_Descent,
	class NonlinearityFunction=bc::Logistic>
struct FeedForward:
		Layer_Base<bc::traits::Integer<1>, ValueType, SystemTag>
{
	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<bc::traits::Integer<1>, ValueType, SystemTag>;

	using self_type = FeedForward<SystemTag, ValueType, Optimizer>;
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

	batched_input_tensor_type x;
	mat w;  //weights
	vec b;  //biases

	mat w_gradients;
	vec b_gradients;

	mat_opt_t w_opt;
	vec_opt_t b_opt;

public:

	FeedForward(int inputs, int outputs):
		parent_type(__func__)
	{
		this->m_input_shape[0] = inputs;
		this->m_output_shape[0] = outputs;
	}

	FeedForward(int outputs):
		parent_type(__func__)
	{
		this->m_output_shape[0] = outputs;
	}

	void init() override
	{
		int inputs = this->m_input_shape[0];
		int outputs = this->m_output_shape[0];

		if (inputs == 0)
			this->m_input_shape = this->prev()->input_shape();

		inputs = this->m_input_shape[0];

		w = mat(outputs, inputs);
		w_opt = mat_opt_t(outputs, inputs);
		w_gradients = mat(outputs, inputs);

		b = vec(outputs);
		b_opt = vec_opt_t(outputs);
		b_gradients = vec(outputs);

		w.randomize(-1, 1);
		b.randomize(-1, 1);
	}

	virtual batched_output_tensor_type forward_propagation(
			const batched_input_tensor_type& x) override
	{
		if (x.inner_shape() == this->x.inner_shape())
			this->x = x;
		else
			this->x = batched_input_tensor_type(x);

		return g(w * this->x + b);
	}

	virtual batched_input_tensor_type back_propagation(
			const batched_output_tensor_type& dy1) override
	{
//		if (!this->prev())
//			return batched_input_tensor_type();

		auto& x = this->prev()->y;
//		batched_output_tensor_type& dy = const_cast<batched_output_tensor_type&>(dy_);
		batched_output_tensor_type dy = dy1 % g.cached_dx(this->y);

		w_gradients -= dy * this->x.t();
		b_gradients -= dy;
		return w.t() * dy;
	}

	virtual void set_learning_rate_hook(double lr) override
	{
		w_opt.set_learning_rate(this->batched_learning_rate());
		b_opt.set_learning_rate(this->batched_learning_rate());
	}

	void update_weights()
	{
		w_opt.update(w, w_gradients);
		b_opt.update(b, b_gradients);
		w_gradients.zero();
		b_gradients.zero();
	}

	virtual void save(Layer_Loader& loader) const override
	{
		loader.save_variable(w, "w");
		loader.save_variable(b, "b");
		w_opt.save(loader, "w_opt");
		b_opt.save(loader, "b_opt");
	}

	virtual void load(Layer_Loader& loader) override
	{
		loader.load_variable(w, "w");
		loader.load_variable(b, "b");
		w_opt.load(loader, "w_opt");
		b_opt.save(loader, "b_opt");
	}
};

template<class SystemTag, class Optimizer=nn_default_optimizer_type>
auto feedforward(SystemTag system_tag, int inputs, int outputs, Optimizer=Optimizer()) {
	using value_type = typename SystemTag::default_floating_point_type;
	return FeedForward<SystemTag, value_type, Optimizer>(inputs, outputs);
}

template<class Optimizer=nn_default_optimizer_type>
auto feedforward(int inputs, int outputs, Optimizer=Optimizer()) {
	using system_tag = BLACKCAT_DEFAULT_SYSTEM_T;
	using value_type = typename system_tag::default_floating_point_type;
	return FeedForward<system_tag, value_type, Optimizer>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
