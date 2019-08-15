/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_LAYER_GREEDYFEEDFORWARD_H_
#define BLACKCAT_NEURALNETWORKS_LAYER_GREEDYFEEDFORWARD_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {


/** Identical to FeedForward layer however it will update its weights during back-prop opposed to update-weights.
 *  This is faster (and only valid for) non-recurrent NeuralNetworks.
 */
template<
	class SystemTag,
	class ValueType>
struct GreedyFeedForward : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;

private:

	ValueType lr = Layer_Base::default_learning_rate;

	mat w;  //weights
	vec b;  //biases

public:

	GreedyFeedForward(BC::size_t inputs, BC::size_t outputs) :
		Layer_Base(__func__, inputs, outputs),
		w(outputs, inputs),
		b(outputs) {
		w.randomize(-2, 2);
		b.randomize(-2, 2);
	}

	template<class Matrix>
	auto forward_propagation(const Matrix& x) {
		return w * x + b;
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		ValueType lr = this->lr / this->batch_size();

		w -= lr * dy * x.t();
		b -= lr * dy;
		return w.t() * dy;
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
GreedyFeedForward<SystemTag, ValueType> greedyfeedforward(SystemTag system_tag, int inputs, int outputs) {
	return GreedyFeedForward<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto greedyfeedforward(SystemTag system_tag, int inputs, int outputs) {
	return GreedyFeedForward<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto greedyfeedforward(int inputs, int outputs) {
	return GreedyFeedForward<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
