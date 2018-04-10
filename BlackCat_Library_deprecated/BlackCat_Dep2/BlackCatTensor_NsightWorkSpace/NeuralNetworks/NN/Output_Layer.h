/*
 * Output_Layer.h
 *
 *  Created on: Oct 18, 2017
 *      Author: joseph
 */

#ifndef OUTPUT_LAYER_H_
#define OUTPUT_LAYER_H_

#include "Layer.h"

class OutputLayer : public Layer {

	//calculates the residual and stores the inputs

	vec forwardPropagation(vec x) {
		bpX.store(std::move(x));

		return bpX.back();
	}
	vec forwardPropagation_express(vec x) {
		return x;
	}
	vec backwardPropagation(vec dy) {
		vec dx = bpX.back() - dy;
		return prev->backwardPropation(dx);
	}
	vec backwardPropagation_ThroughTime(vec dy) {
		vec dx = bpX.back() - dy;
		return prev->backwardPropation(dx);
	}

	//NeuralNetwork update-methods
	void clearBackPropagationStorage() {
		bpX.clear();
	}
	void clearGradientStorage() {/*empty*/}
	void updateGradients() { {/*empty*/}

}

#endif /* OUTPUT_LAYER_H_ */
