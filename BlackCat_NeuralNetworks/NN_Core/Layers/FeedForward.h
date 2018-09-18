/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace NN {

struct FeedForward : public Layer_Base{
public:

	using Layer_Base::lr;	//the learning rate
//	using Layer_Base<derived>::x;


	mat dy;							//error
	mat y;							//outputs
	mat_view x;

	mat w;							//weights
	vec b;							//biases

	mat w_gradientStorage;		//weight gradient storage
	vec b_gradientStorage;		//bias gradient storage


	FeedForward(int inputs, int outputs) :
		Layer_Base(inputs, outputs),
			w(outputs, inputs),
			b(outputs),

			w_gradientStorage(outputs, this->INPUTS),
			b_gradientStorage(outputs)
	{
		w.randomize(-2, 2);
		b.randomize(-1, 1);
		init_storages();
	}

	template<class t> const auto& forward_propagation(const expr::mat<t>& x_) {
		x = mat_view(x_);

		return y = g(w * x + b);
	}
	template<class t> auto back_propagation(const expr::mat<t>& dy_) {
		dy = dy_;

		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return w.t() * dy % gd(x);
	}
	void update_weights() {
		w += w_gradientStorage * lr;
		b += b_gradientStorage * lr;
	}

	void clear_stored_delta_gradients() {
		w_gradientStorage.fill(0);
		b_gradientStorage.fill(0);
	}

	void set_batch_size(int x) {
		y = mat(this->OUTPUTS, x);
		dy = mat(this->OUTPUTS, x);
	}

	void write(std::ofstream& os) {
		w.write(os);
		b.write(os);
	}
	void read(std::ifstream& is) {
		w.read(is);
		b.read(is);
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

#endif /* FEEDFORWARD_CU_ */
