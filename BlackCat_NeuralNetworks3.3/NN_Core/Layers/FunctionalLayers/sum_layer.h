/*
 * sum.h
 *
 *  Created on: Jun 26, 2018
 *      Author: joseph
 */

#ifndef SUM_H_
#define SUM_H_

namespace BC {
namespace NN {

template<class derived>
struct sum_layer : Layer<derived> {

	using Layer<derived>::sum_gradients;		//a function that stores all the gradients within a thread pool
	using Layer<derived>::zero;					//a function that zeros all tensor-parameters
	using Layer<derived>::clear;				//a function that clears back_propagation_lists
	using Layer<derived>::lr;					//the learning rate
	using Layer<derived>::xs;					//the input back_propagation_list (from previous layer)

	omp_unique<vec> w_gradientStorage;		//weight gradient storage
	mat w;

	sum_layer(int inputs)
		: Layer<derived>(inputs),
		  w(this->OUTPUTS)

	{
		w.randomize(-1, 1);
		init_storages();
	}


	template<class f> auto forwardPropagation(const vec<f>& x) {
		return this->next().forwardPropagation(w + x);
	}
	auto backPropagation(const vec& dy) {
		w_gradientStorage() -= dy;

		return this->prev().backPropagation(dy);
	}

	auto forwardPropagation_Express(const vec& x) const {
		return forwardPropagation(x);
	}

	void init_storages() {
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS); var.zero(); });
	}
};

}
}



#endif /* sum_H_ */
