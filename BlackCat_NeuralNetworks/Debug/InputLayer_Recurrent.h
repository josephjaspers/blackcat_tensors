/*
 * InputLayer_Recurrent.h
 *
 *  Created on: Aug 26, 2018
 *      Author: joseph
 */

#ifndef INPUTLAYER_RECURRENT_H_
#define INPUTLAYER_RECURRENT_H_

namespace BC {
namespace NN {

template<class derived>
struct InputLayer_Recurrent : Layer_Base_Recurrent<derived> {

	InputLayer_Recurrent() : Layer_Base_Recurrent<derived>(0) {}

	mat_view* y = new mat_view[this->max_bptt];

	template<class tensor> auto forward_propagation(const tensor& x) {
		y[this->curr_time_stamp] = mat_view(x);
		return this->next().forward_propagation(y[this->curr_time_stamp]);
	}

	template<class tensor> auto forward_propagation_express(const tensor& x) const {
		return this->next().forward_propagation_express(x);
	}

	template<class tensor> auto back_propagation(const tensor& dy) {
		return dy;
	}

	void set_batch_size(int x) {
		this->next().set_batch_size(x);
	}

	void update_weights() {
		this->next().update_weights();
	}
	void clear_stored_delta_gradients() {
		this->next().clear_stored_delta_gradients();
	}
	void write(std::ofstream& os) {
		this->next().write(os);
	}
	void read(std::ifstream& is) {
		this->next().read(is);
	}
	void setLearningRate(fp_type learning_rate) {
		this->next().setLearningRate(learning_rate);
	}

};

}
}



#endif /* INPUTLAYER_RECURRENT_H_ */
