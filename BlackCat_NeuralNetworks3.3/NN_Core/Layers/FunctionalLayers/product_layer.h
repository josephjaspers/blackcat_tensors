/*
 * Product.h
 *
 *  Created on: Jun 26, 2018
 *      Author: joseph
 */

#ifndef PRODUCT_H_
#define PRODUCT_H_

namespace BC {
namespace NN {

template<class derived>
struct product_layer : Layer<derived> {

	using Layer<derived>::sum_gradients;		//a function that stores all the gradients within a thread pool
	using Layer<derived>::zero;					//a function that zeros all tensor-parameters
	using Layer<derived>::clear;				//a function that clears back_propagation_lists
	using Layer<derived>::lr;					//the learning rate
	using Layer<derived>::xs;					//the input back_propagation_list (from previous layer)

	omp_unique<mat> w_gradientStorage;		//weight gradient storage
	mat w;
	bp_list<vec> ys;							//outputs

	product_layer(int inputs)
		: Layer<derived>(inputs),
		  w(this->OUTPUTS, inputs)

	{
		w.randomize(-1, 1);
		init_storages();
	}
	template<class f> auto forwardPropagation(const vec<f>& x) {
		return this->next().forwardPropagation(w * x);
	}
	auto backPropagation(const vec& dy) {
		w_gradientStorage() -= dy * x.t();

		return this->prev().backPropagation(w.t() * dy);
	}

	auto forwardPropagation_Express(const vec& x) const {
		return forwardPropagation(x);
	}

	void init_storages() {
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); var.zero(); });
	}
};

}
}



#endif /* PRODUCT_H_ */
