/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "../../BC_Internals/Layers/Layer_Base.h"

namespace BC {
namespace NN {

struct FeedForward : public Layer_Base{
public:

	using Layer_Base::lr;	//the learning rate

	mat dy;					//error
	mat y;					//outputs
	mat_view x;				//inputs

	mat_shared w;			//weights
	vec_shared b;			//biases

//	mat w_gradientStorage;	//weight gradient storage
//	vec b_gradientStorage;	//bias gradient storage


	FeedForward(int inputs, int outputs) :
		Layer_Base(inputs, outputs),
			w(outputs, inputs),
			b(outputs)//,

//			w_gradientStorage(outputs, this->INPUTS),
//			b_gradientStorage(outputs)
	{	}

	template<class t> const auto& forward_propagation(const expr::mat<t>& x_) {
		x = mat_view(x_);

		return y = g(w * x + b);
	}
	template<class t> auto back_propagation(const expr::mat<t>& dy_) {
		dy = dy_;
		return w.t() * dy % gd(x);
	}
	void update_weights() {
		w -= dy * lr * x.t();
		b -= dy * lr;
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

	auto& activations() { return x; }
	auto& weights()	    { return w; }
	auto& bias()		{ return b; }

	template<class tensor> void set_weight(tensor& workspace) {
		w.internal() = workspace.internal().memptr();		//FIXME w = workspace.internal() //doesn't compile but should
		w.randomize(-2,2);
	}
	template<class tensor> void set_bias(tensor& workspace) {
		b = workspace.internal();
		b.randomize(-1,1);
	}

};
}
}

#endif /* FEEDFORWARD_CU_ */
