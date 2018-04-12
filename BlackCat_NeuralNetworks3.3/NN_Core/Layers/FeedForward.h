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

template<class derived>
struct FeedForward : public Layer<derived> {


public:

	using PARAMTER_TYPE = vec;
	 scal lr = scal(0.03); //fp_type == floating point

	gradient_list<mat> w_gradientStorage;
	gradient_list<vec> b_gradientStorage;

	bp_list<vec> ys;
	auto& xs() { return this->prev().ys(); }

	mat w;
	vec b;

	vec x;
	vec y;

	FeedForward(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS),
			x(inputs),
			y(this->OUTPUTS)
			{

		w.randomize(-4, 4);
		b.randomize(-4, 4);
		init_storages();

	}

	vec forwardPropagation(const vec& x) {
		xs().push_front(x);			//store the inputs

		return this->next().forwardPropagation(g(w * x + b));
	}
	vec backPropagation(const vec& dy) {
		vec x = xs().pop_front();				//load the last input

		w_gradientStorage() -= dy * x.t();		//does work
		b_gradientStorage() -= dy;
		return this->prev().backPropagation(w.t() * dy % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	void updateWeights() {
		ys.clear();
		w_gradientStorage.for_each([&](auto& var) { w += var * lr; });
		b_gradientStorage.for_each([&](auto& var) { b += var * lr; });

		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage.for_each([](auto& var) { var.zero(); });	//gradient lists
		b_gradientStorage.for_each([](auto& var) { var.zero(); });	//gradient list
		ys               .for_each([](auto& var) { var.clear();});	//bp_list

		this->next().clearBPStorage();
	}
	void init_threads(int i) {
		ys.resize(i);
		w_gradientStorage.resize(i);
		b_gradientStorage.resize(i);

		init_storages();

		this->next().init_threads(i);
	}

	void write(std::ofstream& is) {
		is << this->INPUTS << ' ';
		is << this->OUTPUTS << ' ';
		w.write(is);
		b.write(is);
		x.write(is);
		y.write(is);
	}
	void read(std::ifstream& os) {
		os >> this->INPUTS;
		os >> this->OUTPUTS;
		w.read(os);
		b.read(os);
		x.read(os);
		y.read(os);
	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}

	void init_storages() {
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); var.zero(); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			    var.zero(); });

	}

};
}

#endif /* FEEDFORWARD_CU_ */
