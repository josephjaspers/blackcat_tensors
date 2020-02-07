/*
 * Momentum.h
 *
 *  Created on: Dec 3, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_OPTIMIZERS_MOMENTUM_H_
#define BLACKCATTENSORS_NEURALNETWORKS_OPTIMIZERS_MOMENTUM_H_

#include "optimizer_base.h"

namespace bc {
namespace nn {

struct Momentum {

	template<class Tensor>
	struct Optimizer: Optimizer_Base {

		using value_type = typename Tensor::value_type;

		value_type alpha = .9;
		value_type learning_rate = 0.003;

		Tensor momentum;

		template<class... Args>
		Optimizer(Args&&... args):
			momentum(std::forward<Args>(args)...) {
			momentum.zero();
		}

		template<class TensorX, class Gradients>
		void update(TensorX& tensor, Gradients&& delta)
		{
			momentum = alpha * momentum + delta * learning_rate;
			tensor += momentum;
		}

		void set_learning_rate(value_type lr) {
			learning_rate = lr;
		}

		void save(Layer_Loader& loader, std::string name) const {
			loader.save_variable(momentum, name);
		}

		void load(Layer_Loader& loader, std::string name) {
			loader.load_variable(momentum, name);
		}
	};

} momentum;

}
}



#endif /* MOMENTUM_H_ */
