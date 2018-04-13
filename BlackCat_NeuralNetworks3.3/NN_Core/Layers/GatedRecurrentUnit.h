/*
 * GatedRecurrentUnit.h
 *
 *  Created on: Apr 12, 2018
 *      Author: joseph
 */

#ifndef GATEDRECURRENTUNIT_H_
#define GATEDRECURRENTUNIT_H_
#include "Layer.h"
namespace BC {

template<class derived>
struct GRU : public Layer<derived> {

	/*
	 * THIS CLASS HAS NOT BEEN TESTED PELASE TEST ME
	 * 	REMOVE WHEN VALIDATED
	 */

/*
 * 	Note:
 * 		** = point-wise multiply
 * 		*=*  = point-wise multiply and assign
 * 		*  = dot-product
 *
 */

public:

	using Layer<derived>::sum_gradients;		//a function that stores all the gradients within a thread pool
	using Layer<derived>::zero;					//a function that zeros all tensor-parameters
	using Layer<derived>::clear;				//a function that clears back_propagation_lists
	using Layer<derived>::lr;					//the learning rate
	using Layer<derived>::xs;					//the input back_propagation_list (from previous layer)

	gradient_list<mat> wi_gradientStorage; 		//gradient storage input weights
	gradient_list<vec> bi_gradientStorage;		//gradient storage input bias
	gradient_list<mat> ri_gradientStorage;		//gradient storage recurrent input weights

	gradient_list<mat> wf_gradientStorage; 		//gradient storage forget weights
	gradient_list<vec> bf_gradientStorage;		//gradient storage forget bias
	gradient_list<mat> rf_gradientStorage;		//gradient storage recurrent forget weights

	gradient_list<vec> dc;						//cell  state error
	gradient_list<vec> df;						//forget gate error
	gradient_list<vec> di;						//input  gate error

	bp_list<vec> cs;							//storage for cell-states
	bp_list<vec> is;							//storage for inputs gate activation
	bp_list<vec> fs;							//storage for forget gate activation


	bp_list<vec> c;								//cell state
	mat wf, wi;									//weights
	mat rf, ri;									//recurrent weights
	vec bf, bi;									//biases

	GRU(int inputs) :
			Layer<derived>(inputs),
			wf(this->OUTPUTS, this->INPUTS),
			rf(this->OUTPUTS, this->OUTPUTS),
			bf(this->OUTPUTS)
	{

	}

	auto forwardPropagation_Express(const vec& x) const {
		vec f = g(wf * x + rf * c() + bf);
		vec i = g(wi * x + ri * c() + bi);

		c() = c() ** (f + i);

		return this->next().forwardPropagation(c);
	}

	vec forwardPropagation(const vec& x) {
		vec f = g(wf * x + rf * c() + bf);
		vec i = g(wi * x + ri * c() + bi);

		c() *=* (f + i);

		fs().store(x);				//store f
		is().store(i);				//store i
		cs().store(c);

		return this->next().forwardPropagation(c);
	}
	vec backPropagation(const vec& dy) {
		//load relevant tensors from storage
		vec& x = xs().front();
		vec  f = fs().pop_front();
		vec  i = is().pop_front();
		vec	 c = cs().pop_front();		//current time stamps cell state value
		vec& ct = cs().front();			//current time stamp - 1 cell state value (for forget gate)

		//do back prop math
		dc() += (dy + df() * f.t() + di() * i.t());
		df = dc() ** ct ** gd(f);
		di = dc() ** gd(i);

		dc() = dc() *=* f;				//update cell-state error

		//store gradients
		wi_gradientStorage() -= di() * i.t();
		ri_gradientStorage() -= di() * c.t();
		bi_gradientStorage() -= di();

		wf_gradientStorage() -= df() * f.t();
		rf_gradientStorage() -= df() * c.t();
		bf_gradientStorage() -= df();

		vec dx = (wi.t() * di() ** gd(x)) + (wf.t() * df() ** gd(x));
		return this->prev().backPropagation(dx);
	}

	void updateWeights() {
		//sum all the gradients
		wi_gradientStorage.for_each(sum_gradients(wi, lr));
		ri_gradientStorage.for_each(sum_gradients(ri, lr));
		bi_gradientStorage.for_each(sum_gradients(bi, lr));

		wf_gradientStorage.for_each(sum_gradients(wf, lr));
		rf_gradientStorage.for_each(sum_gradients(rf, lr));
		bf_gradientStorage.for_each(sum_gradients(bf, lr));

		this->next().updateWeights();
	}

	void clearBPStorage() {
		cs.for_each(clear);
		fs.for_each(clear);
		is.for_each(clear);

		wi_gradientStorage.for_each(zero);	//gradient list
		ri_gradientStorage.for_each(zero);	//gradient list
		bi_gradientStorage.for_each(zero);	//gradient list

		wf_gradientStorage.for_each(zero);	//gradient list
		rf_gradientStorage.for_each(zero);	//gradient list
		bf_gradientStorage.for_each(zero);	//gradient list


		dc.for_each(zero);
		df.for_each(zero);
		di.for_each(zero);

		this->next().clearBPStorage();
	}
	void init_threads(int i) {
		wi_gradientStorage.resize(i);
		ri_gradientStorage.resize(i);
		bi_gradientStorage.resize(i);

		wf_gradientStorage.resize(i);
		rf_gradientStorage.resize(i);
		bf_gradientStorage.resize(i);

		dc.resize(i);
		df.resize(i);
		di.resize(i);

		init_storages();
	}

	void write(std::ofstream& is) {

	}
	void read(std::ifstream& os) {

	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
	void init_storages() {
		//for each matrix/vector gradient storage initialize to correct dims
		wi_gradientStorage.for_each(zero);
		ri_gradientStorage.for_each(zero);
		bi_gradientStorage.for_each(zero);

		wf_gradientStorage.for_each(zero);
		rf_gradientStorage.for_each(zero);
		bf_gradientStorage.for_each(zero);

		//for each cell-state error initialize to 0
		dc.for_each(zero);
		df.for_each(zero);
		di.for_each(zero);

	}
};
}

#endif /* GATEDRECURRENTUNIT_H_ */
