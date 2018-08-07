/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUT_CU
#define OUTPUT_CU

#include "Layer.h"
namespace BC {
namespace NN {


template<class derived>
struct InputLayer : Layer<derived> {

	InputLayer() : Layer<derived>(0) {}

	mat y;


	template<class tensor> auto forward_propagation(const tensor& x) {
		y = mat(x);
		return this->next().forward_propagation(y);
	}

	template<class tensor> auto forward_propagation_express(const tensor& x) const {
		return this->next().forward_propagation_express(x);
	}

	template<class tensor> auto back_propagation(const tensor& dy) {
		return dy;
	}

	void set_batch_size(int x) {
		this->next().set_batch_size(x);
	}

	void update_weights() {
		this->next().update_weights();
	}
	void clear_stored_delta_gradients() {
		this->next().clear_stored_delta_gradients();
	}
	void write(std::ofstream& os) {
		this->next().write(os);
	}
	void read(std::ifstream& is) {
		this->next().read(is);
	}
	void setLearningRate(fp_type learning_rate) {
		this->next().setLearningRate(learning_rate);
	}

};
}
}
#endif /* FEEDFORWARD_CU_ */
