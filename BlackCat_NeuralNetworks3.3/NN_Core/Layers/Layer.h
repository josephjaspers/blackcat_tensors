/*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "structs/forward_list.h"
#include "structs/omp_unique.h"
#include "Layer_Utility_Functions.h"
namespace BC {
namespace NN {

template<class derived>
class Layer {
public:

	scal lr;

	auto& xs() { return this->prev().ys(); }
	const int INPUTS;
	const int OUTPUTS = static_cast<derived&>(*this).hasNext() ? this->next().INPUTS : INPUTS;

	Layer(int inputs) : INPUTS(inputs) {
		lr = .03;
	}
	auto& next() { return static_cast<derived&>(*this).next(); }
	auto& prev() { return static_cast<derived&>(*this).prev(); }
	const auto& next() const { return static_cast<derived&>(*this).next(); }
	const auto& prev() const { return static_cast<derived&>(*this).prev(); }

	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};

}
}



#endif /* LAYER_H_ */
