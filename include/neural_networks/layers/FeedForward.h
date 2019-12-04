/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class Optimizer=Stochastic_Gradient_Descent>
struct FeedForward:
		public Layer_Base<FeedForward<SystemTag, ValueType, Optimizer>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<FeedForward<SystemTag, ValueType, Optimizer>>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using optimizer_type = Optimizer;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using mat_opt_t = typename Optimizer::template Optimizer<mat>;
	using vec_opt_t = typename Optimizer::template Optimizer<vec>;

	using greedy_evaluate_delta = std::true_type;

private:

	ValueType lr = FeedForward::default_learning_rate;

	mat w;  //weights
	vec b;  //biases

	mat w_gradients;
	vec b_gradients;

	mat_opt_t w_opt;
	vec_opt_t b_opt;


public:

	FeedForward(BC::size_t inputs, BC::size_t outputs):
		parent_type(__func__, inputs, outputs),
		w(outputs, inputs),
		b(outputs),
		w_gradients(outputs, inputs),
		b_gradients(outputs),
		w_opt(w.get_shape()),
		b_opt(b.get_shape())
	{
		w.randomize(-2, 2);
		b.randomize(-2, 2);
		w_gradients.zero();
		b_gradients.zero();
	}

	template<class Matrix>
	auto forward_propagation(const Matrix& x)
	{
		return w * x + b;
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy)
	{
		w_gradients -= dy  * x.t();
		b_gradients -= dy;
		return w.t() * dy;
	}

	void set_learning_rate(value_type lr)
	{
		parent_type::set_learning_rate(lr);
		w_opt.set_learning_rate(lr);
		b_opt.set_learning_rate(lr);
	}

	void update_weights()
	{
		w_opt.update(w, w_gradients);
		b_opt.update(b, b_gradients);
		w_gradients.zero();
		b_gradients.zero();
	}

	void save(Layer_Loader& loader)
	{
		loader.save_variable(w, "w");
		loader.save_variable(b, "b");
	}

	void load(Layer_Loader& loader)
	{
		loader.load_variable(w, "w");
		loader.load_variable(b, "b");
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
