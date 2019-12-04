/*
 * Momentum.h
 *
 *  Created on: Dec 3, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_OPTIMIZERS_MOMENTUM_H_
#define BLACKCATTENSORS_NEURALNETWORKS_OPTIMIZERS_MOMENTUM_H_

namespace BC {
namespace nn {

struct Momentum {

		template<class Tensor>
		struct Optimizer {

		using value_type = typename Tensor::value_type;

		value_type alpha = .9;
		value_type learning_rate = 0.003;

		Tensor momentum;

		template<class... Args>
		Optimizer(Args&&... args):
			momentum(std::forward<Args>(args)...) {
			momentum.zero();
		}

		template<class TensorX, class Gradeients>
		void update(TensorX& tensor, Gradeients&& delta)
		{
			momentum = alpha * momentum + delta * learning_rate;
			tensor += momentum;
		}

		void set_learning_rate(value_type lr) {
			learning_rate = lr;
		}
	};
} momentum;

struct Stochastic_Gradient_Descent {

	template<class ValueType>
	struct Optimizer {

		using value_type = BC::traits::conditional_detected_t<
				BC::traits::query_value_type, ValueType, ValueType>;

		value_type learning_rate = 0.003;

		template<class... Args>
		Optimizer(Args&&...) {}

		template<class TensorX, class Gradeients>
		void update(TensorX& tensor, Gradeients&& delta) {
			tensor += learning_rate * delta;
		}

		void set_learning_rate(value_type lr) {
			learning_rate = lr;
		}
	};
} sgd;

}
}



#endif /* MOMENTUM_H_ */
