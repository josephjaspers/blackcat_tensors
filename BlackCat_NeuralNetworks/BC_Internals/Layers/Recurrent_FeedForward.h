/*
 * Recurrent_Recurrent_FeedForward.h
 *
 *  Created on: Sep 24, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_LAYERS_RECURRENT_Recurrent_FeedForward_H_
#define BC_INTERNALS_LAYERS_RECURRENT_Recurrent_FeedForward_H_

#include "Layer_Base_Recurrent.h"
#include "Utility.h"

namespace BC {
namespace NN {

struct Recurrent_FeedForward : public Layer_Base_Recurrent {

	using Layer_Base_Recurrent::lr;	//learning rate
	using Layer_Base_Recurrent::t;

	cube y;          //outputs
	cube_view x;     //inputs

	mat dy;          //error
	mat w;           //weights
	vec b;           //biases

	mat dw;			 //delta weights
	vec db;			 //delta biases

	Recurrent_FeedForward(int inputs, int outputs) :
		Layer_Base_Recurrent(inputs, outputs),
			w(outputs, inputs),
			b(outputs)
	{
		w.randomize(-1, 1);
		b.randomize(-1, 1);
	}

	template<class T>
	const auto& forward_propagation(const expr::mat<T>& x_) {
		x[t] = x_;

		y = g(w * x[t] + b);

		t++;	//increment current_time
		return y;
	}

	template<class T>
	auto back_propagation(const expr::mat<T>& dy_) {
		dy = dy_;

		auto dx = w.t() * dy % gd(x[t]);
		return dx;
	}
	template<class T>
	auto back_propagation_through_time(const expr::mat<T>& dy_) {
		return this->back_propagation(dy_);
	}

	void cache_gradients() {
		t--;
		dw -= dy * lr * x[t].t();
		db -= dy * lr;
	}
	void update_weights() {
		b += db;
		w += dw;
	}

	void set_batch_size(int batch_sz) {
		y = cube(this->OUTPUTS, batch_sz, this->get_max_bptt_length());
		x = cube_view(this->INPUTS, batch_sz, this->get_max_bptt_length());

		dy = mat(this->OUTPUTS, batch_sz);
	}

	auto& inputs()  { return x; }
	auto& outputs() { return y; }
	auto& deltas()  { return dy;}
	auto& weights()	{ return w; }
	auto& bias()	{ return b; }
};
}
}

#endif /* BC_INTERNALS_LAYERS_RECURRENT_Recurrent_FeedForward_H_ */
