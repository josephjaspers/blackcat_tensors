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

template<class derived>
struct FeedForward : public Layer_Base<derived> {
public:

	using Layer_Base<derived>::lr;	//the learning rate
//	using Layer_Base<derived>::x;


	mat dy;							//error
//	mat y;							//outputs

	mat w;							//weights
	vec b;							//biases

	mat w_gradientStorage;		//weight gradient storage
	vec b_gradientStorage;		//bias gradient storage


	mat x;
	mat y;

	FeedForward(int inputs) :
		Layer_Base<derived>(inputs),
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
		y = g(w * x + b);

		return this->next().forward_propagation(y);
	}
	template<class t> auto back_propagation(const expr::mat<t>& dy_) {
		dy = dy_;

		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return this->prev().back_propagation(w.t() * dy % gd(x));
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

	void init_input_view(vec& workspace, int& offset, int batch_size) {
		int n_inputs  = this->inputs();																			//inputs are allocated from a single block of memory
		int input_sz = n_inputs* batch_size;
		auto input_slice = workspace[{offset, offset + input_sz}]; 				//like Python ranged slice ws[offset:offset+ws_sz]
		auto input_shaped = reshape(input_slice)(this->outputs(), batch_size);	//reshape is curreid function
		x = mat_view(input_shaped);

		offset += input_sz;

		int n_outputs = this->outputs();
		int output_sz  = n_outputs * batch_size;
		auto output_slice = workspace[{offset, offset + output_sz}]; 				//like Python ranged slice ws[offset:offset+ws_sz]
		auto output_shaped = reshape(input_slice)(this->outputs(), batch_size);	//reshape is curreid function
		y = mat_view(output_shaped);

		this->next().init_input_view(workspace, offset, batch_size);
	}

};
}
}

#endif /* FEEDFORWARD_CU_ */
