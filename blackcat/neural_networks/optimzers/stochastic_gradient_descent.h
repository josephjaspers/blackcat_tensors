/*
 * Stochastic_Gradient_Descent.h
 *
 *  Created on: Dec 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_SGD_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_SGD_H_

#include "optimizer_base.h"

namespace bc {
namespace nn {

struct Stochastic_Gradient_Descent {

	template<class ValueType>
	struct Optimizer : Optimizer_Base {

		using value_type = bc::traits::conditional_detected_t<
				bc::traits::query_value_type, ValueType, ValueType>;

		value_type learning_rate = 0.003;

		template<class... Args>
		Optimizer(Args&&...) {}

		template<class TensorX, class Gradients>
		void update(TensorX& tensor, Gradients&& delta) {
			tensor += learning_rate * delta;
		}

		void set_learning_rate(value_type lr) {
			learning_rate = lr;
		}
	};

} sgd;

}
}

#endif
