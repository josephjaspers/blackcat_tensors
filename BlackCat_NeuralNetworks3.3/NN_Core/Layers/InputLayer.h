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

	vec y;


	template<class T> auto forwardPropagation(const T& x) {
		y = vec(x);
		return this->next().forwardPropagation(x);
	}

	template<class T> auto forwardPropagation_Express(const T& x) const {
		return this->next().forwardPropagation_Express(x);
	}

	template<class T> auto backPropagation(const T&& dy) {
		return dy;
	}
	void set_omp_threads(int i) {
		this->next().set_omp_threads(i);
	}
	void updateWeights() {
		this->next().updateWeights();
	}
	void clearBPStorage() {
		this->next().clearBPStorage();
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
