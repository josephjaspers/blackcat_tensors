/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_LAYER_GREEDYFEEDFORWARD_H_
#define BLACKCAT_NEURALNETWORKS_LAYER_GREEDYFEEDFORWARD_H_

#include "FeedForward.h"

namespace BC {
namespace nn {


/** Identical to FeedForward layer however it will update its weights during back-prop opposed to update-weights.
 *  This is faster (and only valid for) non-recurrent NeuralNetworks.
 */
template<
	class SystemTag,
	class ValueType>
struct GreedyFeedForward: FeedForward<SystemTag, ValueType> {

	GreedyFeedForward(BC::size_t inputs, BC::size_t outputs):
		FeedForward<SystemTag, ValueType>(inputs, outputs) {}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		auto& w = this->get_weight();
		auto& b = this->get_bias();
		auto lr = this->get_learning_rate();

		lr /= this->batch_size();
		w -= lr * dy * x.t();
		b -= lr * dy;
		return w.t() * dy;
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
