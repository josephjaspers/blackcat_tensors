/*
 * addition_layer.h
 *
 *  Created on: Jul 29, 2018
 *      Author: joseph
 */

#ifndef SIGMOID_LAYER_H_
#define SIGMOID_LAYER_H_

#include "Layer.h"
#include <mutex>

namespace BC {
namespace NN {

template<class derived>
struct sigmoid : public Layer<derived> {


	mat x;

	sigmoid(int inputs) :
			Layer<derived>(inputs)
			{}
	template<class e> auto forward_propagation(const expr::mat<e>& x_) {
		x = x_;
		return this->next().forward_propagation(g(x));
	}
	template<class e> auto back_propagation(const expr::mat<e>& dy_) {
		return this->prev().back_propagation(dy_ % gd(x));
	}
	template<class e> auto forward_propagation_express(const expr::mat<e>& x) const {
		return this->next().forward_propagation(g(x));
	}

	void update_weights() {
		this->next().update_weights();
	}

	void clear_stored_delta_gradients() {
		this->next().clear_stored_delta_gradients();
	}

	void set_batch_size(int x) {
		this->x = mat(this->INPUTS, x);
		this->next().set_batch_size(x);
	}

	void write(std::ofstream& os) {
		this->next().write(os);
	}
	void read(std::ifstream& is) {
		this->next().read(is);
	}
	void init_storages() {
		this->next().init_storages();
	}

};
}
}


#endif /* ADDITION_LAYER_H_ */
