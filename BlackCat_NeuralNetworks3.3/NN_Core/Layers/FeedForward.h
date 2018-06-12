/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer.h"
#include <mutex>

namespace BC {
namespace NN {


template<class derived>
struct FeedForward : public Layer<derived> {

public:

	using Layer<derived>::sum_gradients;		//a function that stores all the gradients within a thread pool
	using Layer<derived>::zero;					//a function that zeros all tensor-parameters
	using Layer<derived>::clear;				//a function that clears back_propagation_lists
	using Layer<derived>::lr;					//the learning rate
	using Layer<derived>::xs;					//the input back_propagation_list (from previous layer)

	omp_unique<mat> w_gradientStorage;		//weight gradient storage
	omp_unique<vec> b_gradientStorage;		//bias gradient storage

	bp_list<vec> ys;							//outputs

	mat w;										//weights
	vec b;										//biases

	FeedForward(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS)
	{

		w.randomize(-1, 1);
		b.randomize(-1, 1);
		init_storages();
	}

	auto forwardPropagation(const vec& x) {
		vec y = g(w * x + b);

		ys().push(std::move(y));
		return this->next().forwardPropagation(ys().first());
	}
	auto backPropagation(const vec& dy) {
		ys().rm_front();
		vec& x = xs().first();				//load the last input

		w_gradientStorage() -= dy * x.t();
		b_gradientStorage() -= dy;

		return this->prev().backPropagation(w.t() * dy % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	void updateWeights() {
		w_gradientStorage.for_each(sum_gradients(w, lr));
		b_gradientStorage.for_each(sum_gradients(b, lr));

		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage.for_each(zero);	//gradient lists
		b_gradientStorage.for_each(zero);	//gradient list
		ys.for_each(clear);					//bp_list

		this->next().clearBPStorage();
	}
	void set_omp_threads(int i) {
		ys.resize(i);
		w_gradientStorage.resize(i);
		b_gradientStorage.resize(i);

		init_storages();

		this->next().set_omp_threads(i);
	}

	void write(std::ofstream& os) {
		w.write(os);
		b.write(os);

		this->next().write(os);
	}
	void read(std::ifstream& is) {
		std::string tmp;
		w.read(is);
		b.read(is);

		this->next().read(is);
	}

	void init_storages() {
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); var.zero(); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			    var.zero(); });
	}

};
}
}

#endif /* FEEDFORWARD_CU_ */
