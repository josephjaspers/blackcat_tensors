 /*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

//#include "Layer_Utility_Functions.h"
namespace BC {
namespace NN {

using NN_Abreviated_Functions::g;
using NN_Abreviated_Functions::gd;


template<class derived>
class Layer_Base {
public:

	const int INPUTS;
	const int OUTPUTS;
	scal lr;

	Layer_Base(int inputs, int outputs)
		: INPUTS(inputs),
		  OUTPUTS(outputs),
		  lr(fp_type(.03)) {}

	int inputs() const { return INPUTS; }
	int outputs() const { return OUTPUTS; }

	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
	}
};

}
}



#endif /* LAYER_H_ */
