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
	class ValueType>
struct FeedForward:
		public Layer_Base<FeedForward<SystemTag, ValueType>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<FeedForward<SystemTag, ValueType>>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;

private:

	ValueType lr = FeedForward::default_learning_rate;

	mat w;  //weights
	vec b;  //biases

	mat w_gradients;
	vec b_gradients;

public:

	FeedForward(BC::size_t inputs, BC::size_t outputs):
		parent_type(__func__, inputs, outputs),
		w(outputs, inputs),
		b(outputs),
		w_gradients(outputs, inputs),
		b_gradients(outputs)
	{
		w.randomize(-2, 2);
		b.randomize(-2, 2);
		w_gradients.zero();
		b_gradients.zero();
	}

	template<class Matrix>
	auto forward_propagation(const Matrix& x) {
		return w * x + b;
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		w_gradients -= dy  * x.t();
		b_gradients -= dy;
		return w.t() * dy;
	}

	void update_weights() {
		ValueType lr = this->lr / this->batch_size();
		w += w_gradients * lr;
		b += b_gradients * lr;
		w_gradients.zero();
		b_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
		loader.save_variable(b, "b");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
		loader.load_variable(b, "b");
	}
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
FeedForward<SystemTag, ValueType> feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto feedforward(int inputs, int outputs) {
	return FeedForward<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
