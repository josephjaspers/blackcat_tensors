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

template<class Tensor>
struct Momentum {

	using value_type = typename Tensor::value_type;

	value_type alpha = .9;
	value_type learning_rate = 0.003;

	Tensor momentum;

	template<class... Args>
	Momentum(Args&&... args):
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


template<class ValueType>
struct Stochastic_Gradient_Descent {

	using value_type = ValueType;
	value_type learning_rate = 0.003;

	template<class TensorX, class Gradeients>
	void update(TensorX& tensor, Gradeients&& delta) {
		tensor += learning_rate * delta;
	}
};

}
}



#endif /* MOMENTUM_H_ */
