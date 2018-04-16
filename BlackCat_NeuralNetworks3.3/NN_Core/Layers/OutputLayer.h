/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "Layer.h"

namespace BC {
namespace NN {

template<class derived>
struct OutputLayer : Layer<derived> {

	using Layer<derived>::xs;

public:

	OutputLayer(int inputs) : Layer<derived>(inputs) {
	}

	vec& forwardPropagation(const vec& x) {
		xs().push(x);
		return xs().first();
	}
	vec forwardPropagation_Express(const vec& x) const {
		return x;
	}
	vec backPropagation(const vec& y) {
		vec& x = xs().first();
		return this->prev().backPropagation(x - y);
	}

	void set_omp_threads(int i) {
	}

	void updateWeights() {}
	void clearBPStorage() {}
	void write(std::ofstream& is) {
	}
	void read(std::ifstream& os) {
	}
	void setLearningRate(fp_type learning_rate) {
		return;
	}

};

}
}



#endif /* FEEDFORWARD_CU_ */
