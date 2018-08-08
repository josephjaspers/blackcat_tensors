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

template<class derived, int tensor_dimension = 1>
class Layer {
public:

	Scalar lr;

	const int INPUTS;
	const int OUTPUTS;

	Layer(int inputs)
		: INPUTS(inputs),
		  OUTPUTS(as_derived().hasNext() ? next().INPUTS : INPUTS),
		  lr(fp_type(.03)) {}

	const auto& as_derived() const { return static_cast<const derived&>(*this); }

	const auto& next() const { return static_cast<const derived&>(*this).next(); }
		  auto& next() 		 { return static_cast<derived&>(*this).next(); }
	const auto& prev() const { return static_cast<const derived&>(*this).prev(); }
		  auto& prev() 		 { return static_cast<derived&>(*this).prev(); }

	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};

}
}



#endif /* LAYER_H_ */
