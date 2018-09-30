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

static const BC::internal::Shape<0> NULL_TENSOR = BC::internal::Shape<0>();

//template<class derived>
class Layer_Base {
public:

	static constexpr bool is_recurrent = false;

	const int INPUTS;
	const int OUTPUTS;
	int BATCH_SIZE;
	scal lr;

	Layer_Base(int inputs, int outputs)
		: INPUTS(inputs),
		  OUTPUTS(outputs),
		  BATCH_SIZE(1),
		  lr(fp_type(.03)) {}

	int numb_inputs() const { return INPUTS; }
	int numb_outputs() const { return OUTPUTS; }
	int batch_size()   const { return BATCH_SIZE; }
	void set_batch_size(int bs) {
		BATCH_SIZE = bs;
	}
	void set_learning_rate(fp_type learning_rate) {
		lr = learning_rate;
	}
};

}
}



#endif /* LAYER_H_ */
