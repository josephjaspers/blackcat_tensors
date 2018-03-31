/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUT_CU
#define OUTPUT_CU

#include "Defaults.h"
#include "Layer.h"
namespace BC {

template<class derived>
struct InputLayer : Layer<derived> {

	InputLayer() : Layer<derived>(0) {}

	template<class T> auto forwardPropagation(const T& x) {
		return this->next().forwardPropagation(x);
	}

	template<class T> auto forwardPropagation_Express(const T& x) const {
		return this->next().forwardPropagation_Express(x);
	}

	template<class T> auto backPropagation(const T& dy) {
		return dy;
	}

	void train(const vec& x, const vec& y) {
		this->next().train(x, y);
	}

	void updateWeights() {
		this->next().updateWeights();
	}
	void clearBPStorage() {
		this->next().clearBPStorage();
	}

	void write(std::ofstream& is) {
		this->next().write(is);
	}
	void read(std::ifstream& os) {
		this->next().read(os);
	}
	void setLearningRate(fp_type learning_rate) {
		this->next().setLearningRate(learning_rate);
	}

};
}




#endif /* FEEDFORWARD_CU_ */
