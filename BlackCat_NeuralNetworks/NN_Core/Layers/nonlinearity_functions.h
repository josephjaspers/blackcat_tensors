/*
 * nonlineariy_functions.h
 *
 *  Created on: Sep 14, 2018
 *      Author: joseph
 */

#ifndef NN_CORE_LAYERS_NONLINEARITY_FUNCTIONS_H_
#define NN_CORE_LAYERS_NONLINEARITY_FUNCTIONS_H_

namespace BC {
namespace NN {

struct sigmoid {

	template<class tensor>
	auto operator() (const tensor& t) const {
		 return BC::NN_Functions::sigmoid(t);
	}

	template<class tensor>
	auto d(const tensor& t) const {
		 return BC::NN_Functions::sigmoid_deriv(t);
	}
};

}
}



#endif /* NN_CORE_LAYERS_NONLINEARITY_FUNCTIONS_H_ */
