/*
 * Layer.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef LAYER_CU_
#define LAYER_CU_

#include "BlackCat_Tensors.h"
#include "BlackCat_TensorFunctions.cu"
#include "Defaults.cu"

namespace BC {
using namespace NN_Abreviated_Functions;

static const Scalar<fp_type> lr = 0.03; //fp_type == floating point


template<class derived>
class Layer {

		  derived& asBase() 	  { return static_cast<	     derived&>(*this); }
	const derived& asBase() const { return static_cast<const derived&>(*this); }

public:


	template<class AUTO> auto forwardPropagation		(AUTO param) 	   { return asBase().forwardPropagation(param); }
	template<class AUTO> auto forwardPropagation_Express(AUTO param) const { return asBase().forwardPropagation(param); }

	template<class AUTO> auto backPropagation			 (AUTO param) { return asBase().forwardPropagation(param); }
	template<class AUTO> auto backPropagation_ThroughTime(AUTO param) { return asBase().forwardPropagation(param); }

	void updateWeights()  { asBase().updateWeights();  }
	void clearBPStorage() { asBase().clearBPStorage(); }


	void read(std::ofstream& os) { asBase().read(os); }
	void write(std::ifstream& is) { asBase().write(is); }

};

}

#endif /* LAYER_CU_ */
