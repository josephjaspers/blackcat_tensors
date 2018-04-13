/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef Recurrent_Unit
#define Recurrent_Unit

#include "Layer.h"
#include <mutex>

namespace BC {

template<class derived>
struct Recurrent : public Layer<derived> {

	/*
	 * THIS CLASS HAS NOT BEEN TESTED PELASE TEST ME
	 * 	REMOVE WHEN VALIDATED
	 */

public:

	using Layer<derived>::sum_gradients;
	using Layer<derived>::zero;

	using Layer<derived>::lr;

	gradient_list<mat> w_gradientStorage; 		//gradient storage weights
	gradient_list<mat> r_gradientStorage;		//gradient storage recurrent weights
	gradient_list<vec> b_gradientStorage;		//gradienst storage bias
	gradient_list<vec> dc;

	bp_list<vec> ys;							//storage for outputs
	auto& xs() { return this->prev().ys(); }	//get the storage for inputs

	mat w;
	mat r;
	vec b;

	Recurrent(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, this->INPUTS),
			r(this->OUTPUTS, this->OUTPUTS),
			b(this->OUTPUTS)
		{
		r.randomize(-4, 0);
		w.randomize(-4, 4);
		b.randomize(-4, 4);
		init_storages();

	}


	vec forwardPropagation(const vec& x) {
		xs().push_front(x);	//store the inputs

		if (ys().isEmpty())
			return this->next().forwardPropagation(g(w * x + b));
		else {
			vec& y = ys().front();
			return this->next().forwardPropagation(g(w * x + r * y + b));
		}

	}
	vec backPropagation(const vec& dy) {
		vec& x = xs().front();				//load the last input
		vec  y = ys().pop_front();			//load last and remove

		w_gradientStorage() -= dy   * x.t();
		r_gradientStorage() -= dc() * y.t();		//dc() is a function call as each thread has its own cell state error
		b_gradientStorage() -= dy;

		dc() += dy;

		return this->prev().backPropagation(w.t() * dy % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		if (ys().isEmpty())
			return this->next().forwardPropagation(g(w * x + b));
		else {
			auto& y = ys().front();
			return this->next().forwardPropagation(g(w * x + r * y + b));
		}
	}

	void updateWeights() {
		//sum all the gradients
		w_gradientStorage.for_each(sum_gradients(w, lr));
		r_gradientStorage.for_each(sum_gradients(r, lr));
		b_gradientStorage.for_each(sum_gradients(b, lr));

		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage.for_each(zero);	//gradient list
		r_gradientStorage.for_each(zero);	//gradient list
		b_gradientStorage.for_each(zero);	//gradient list

		dc.for_each([](auto& var) { var.zero(); }); 	//gradient list
		ys.for_each([](auto& var) { var.clear();});		//bp_list

		this->next().clearBPStorage();
	}
	void init_threads(int i) {
		ys.resize(i);
		dc.resize(i);
		w_gradientStorage.resize(i);
		b_gradientStorage.resize(i);

		init_storages();
	}

	void write(std::ofstream& is) {
		is << this->INPUTS << ' ';
		is << this->OUTPUTS << ' ';
		w.write(is);
		r.write(is);
		b.write(is);
	}
	void read(std::ifstream& os) {
		os >> this->INPUTS;
		os >> this->OUTPUTS;

		w.read(os);
		r.read(os);
		b.read(os);
	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
	void init_storages() {
		//for each matrix/vector gradient storage initialize to correct dims
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS);  var.zero(); });
		r_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->OUTPUTS); var.zero(); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			     var.zero(); });

		//for each cell-state error initialize to 0
		dc.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
	}
};
}


#endif /* FEEDFORWARD_CU_ */
