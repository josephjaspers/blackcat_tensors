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

	using Layer<derived>::clear;

	InputLayer() : Layer<derived>(0) {}

	bp_list<vec> ys;


	template<class T> auto forwardPropagation(const T& x) {
		ys().push(x);
		return this->next().forwardPropagation(x);
	}

	template<class T> auto forwardPropagation_Express(const T& x) const {
		return this->next().forwardPropagation_Express(x);
	}

	template<class T> vec backPropagation(const T&& dy) {
		ys().pop();
		return dy;
	}
	void updateWeights() {
		this->next().updateWeights();
	}
	void clearBPStorage() {
		ys.for_each(clear);
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
}




#endif /* FEEDFORWARD_CU_ */
