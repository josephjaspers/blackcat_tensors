/*
 * Adam.h
 *
 *  Created on: Dec 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_ADAM_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_OPTIMIZERS_ADAM_H_

#include "Optimizer_Base.h"

namespace BC {
namespace nn {

struct Adam {

	template<class Tensor>
	struct Optimizer;

} adam;


template<class Tensor>
struct Adam::Optimizer: Optimizer_Base {

	using value_type = typename Tensor::value_type;
	using system_tag = typename Tensor::system_tag;

	value_type alpha = BC::nn::default_learning_rate;
	value_type beta_1 = 0.9;
	value_type beta_2 = 0.999;
	value_type epsilon = 1e-8;
	value_type time_stamp = 0;

	Tensor m_t;
	Tensor v_t;

	template<class... Args>
	Optimizer(Args&&... args):
		m_t(std::forward<Args>(args)...),
		v_t(std::forward<Args>(args)...) {

		m_t.zero();
		v_t.zero();
	}

	template<class TensorX, class Gradients>
	void update(TensorX& tensor, Gradients&& delta)
	{
		time_stamp++;
		m_t = beta_1 * m_t + (1-beta_1) * delta;
		v_t = beta_2 * v_t + (1-beta_2) * BC::pow2(delta);

		auto m_cap = m_t/(1-(BC::pow(beta_1, time_stamp)));
		auto v_cap = v_t/(1-(BC::pow(beta_2, time_stamp)));

		tensor += (alpha*m_cap)/(BC::sqrt(v_cap)+epsilon);
	}


	void set_learning_rate(value_type lr) {
		alpha = lr;
	}

	void save(Layer_Loader& loader, std::string name) {
		//TODO add support for loader saving primitives
	}

	void load(Layer_Loader& loader, std::string name) {
		//TODO add support for loader loading primitives
	}
};

}
}




#endif /* ADAM_H_ */
