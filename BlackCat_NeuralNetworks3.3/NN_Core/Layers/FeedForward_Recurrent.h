/*
 * FeedForward_Recurrent.h
 *
 *  Created on: Aug 26, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_RECURRENT_H_
#define FEEDFORWARD_RECURRENT_H_

#include "Layer.h"

namespace BC {
namespace NN {

template<class derived>
struct FeedForward_Recurrent : public Layer_Base_Recurrent<derived> {
public:

	using Layer_Base_Recurrent<derived>::lr;	//the learning rate
	using Layer_Base_Recurrent<derived>::x;
	using Layer_Base_Recurrent<derived>::curr_timestamp;
	mat w_gradientStorage;		//weight gradient storage
	vec b_gradientStorage;		//bias gradient storage

	mat dy;							//error
	cube y;							//outputs

	mat w;							//weights
	vec b;							//biases

	FeedForward(int inputs) :
		Layer_Base_Recurrent<derived>(inputs),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS),

			w_gradientStorage(this->OUTPUTS, this->INPUTS),
			b_gradientStorage(this->OUTPUTS)
	{
		w.randomize(-2, 2);
		b.randomize(-1, 1);
		init_storages();
	}

	template<class t> auto forward_propagation(const expr::mat<t>& x) {
		y[curr_timestamp] = g(w * x + b);

		return this->next().forward_propagation(y);
	}
	template<class t> auto back_propagation(const expr::mat<t>& dy_) {
		dy = dy_;

		w_gradientStorage -= dy * x()[curr_time_stamp].t();
		b_gradientStorage -= dy;

		return this->prev().back_propagation(w.t() * dy % gd(x()));
	}
	template<class t> auto forward_propagation_tess(const expr::mat<t>& x) const {
		return this->next().forward_propagation_tess(g(w * x + b));
	}

	void update_weights() {
		w += w_gradientStorage * lr;
		b += b_gradientStorage * lr;

		this->next().update_weights();
	}

	void clear_stored_delta_gradients() {
		w_gradientStorage.fill(0);
		b_gradientStorage.fill(0);

		this->next().clear_stored_delta_gradients();
	}

	void set_batch_size(int x) {
		y = mat(this->OUTPUTS, x);
		dy = mat(this->OUTPUTS, x);

		this->next().set_batch_size(x);
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

		w_gradientStorage.fill(0);
		b_gradientStorage.fill(0);
	}

};
}
}




#endif /* FEEDFORWARD_RECURRENT_H_ */
