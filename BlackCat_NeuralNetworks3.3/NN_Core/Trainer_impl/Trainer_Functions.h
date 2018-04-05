/*
 * Trainer_Functions.h
 *
 *  Created on: Apr 5, 2018
 *      Author: joseph
 */

#ifndef TRAINER_FUNCTIONS_H_
#define TRAINER_FUNCTIONS_H_

namespace BC {
namespace NN {

class fp {

	template<class NN, class... params, class param>
	auto operator () (NN& network, param& f, params&... p) {
		return network.forwardPropagation(f);
	}

};
class bp {
	template<class NN, class... params, class param>
	auto operator () (NN& network, param& f, params&... p) {
		return network.backPropagation(f);
	}
};
class update {
	template<class NN, class... params>
	auto operator () (NN& network, params&... p) {
		return network.updateWeights();
	}
};
class clear {
	template<class NN, class... params>
	auto operator () (NN& network, params&... p) {
		return network.clearBPStorage();
	}
};

}}



#endif /* TRAINER_FUNCTIONS_H_ */
