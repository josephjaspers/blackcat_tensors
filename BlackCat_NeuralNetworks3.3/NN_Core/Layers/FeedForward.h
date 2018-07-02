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

	using Layer<derived>::lr;					//the learning rate

	mat w_gradientStorage;		//weight gradient storage
	vec b_gradientStorage;		//bias gradient storage

	vec y;							//outputs

	mat w;							//weights
	vec b;							//biases
	vec& x = this->prev().y;

	FeedForward(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS),
			y(this->OUTPUTS),
			w_gradientStorage(this->OUTPUTS, this->INPUTS),
			b_gradientStorage(this->OUTPUTS)
	{
		w.randomize(-1, 1);
		b.randomize(-1, 1);
		init_storages();
	}

	auto forwardPropagation(const vec& x) {
		y = g(w * x + b);
		return this->next().forwardPropagation(y);
	}
	auto backPropagation(const vec& dy) {
		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return this->prev().backPropagation(w.t() * dy % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	void updateWeights() {
		w += w_gradientStorage * lr;
		b += b_gradientStorage * lr;

		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage = 0;	//gradient lists
		b_gradientStorage = 0; //.for_each(zero);	//gradient list

		this->next().clearBPStorage();
	}

	void write(std::ofstream& os) {
		w.write(os);
		b.write(os);

		this->next().write(os);
	}
	void read(std::ifstream& is) {
		w.read(is);
		b.read(is);

		this->next().read(is);
	}
	void init_storages() {
		w_gradientStorage = mat(this->OUTPUTS, this->INPUTS);
		b_gradientStorage = vec(this->OUTPUTS);

		w_gradientStorage = 0;
		b_gradientStorage = 0;
	}

};
}
}

#endif /* FEEDFORWARD_CU_ */
