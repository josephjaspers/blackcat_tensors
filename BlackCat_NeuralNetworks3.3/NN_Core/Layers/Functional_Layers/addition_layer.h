/*
 * addition_layer.h
 *
 *  Created on: Jul 29, 2018
 *      Author: joseph
 */

#ifndef ADDITION_LAYER_H_
#define ADDITION_LAYER_H_


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
namespace NN {

template<class derived>
struct Addition : public Layer<derived> {
public:

	using Layer<derived>::lr;
	vec w_gradientStorage;
	mat y;

	vec w;
	mat& x = this->prev().y;

	Addition(int inputs) :
			Layer<derived>(inputs),
			w(this->inputs),
			w_gradientStorage(this->INPUTS)
	{
		b.randomize(-1, 1);
		init_storages();
	}

	template<class expr> auto forward_propagation(const f_mat<expr>& x) {
		auto y = x + w;
		return this->next().forward_propagation(y);
	}
	template<class expr> auto back_propagation(const f_mat<expr>& dy_) {
		b_gradientStorage -= dy;
		return this->prev().back_propagation(dy_);
	}
	template<class expr> auto forward_propagation_express(const f_mat<expr>& x) const {
		auto y = x + w;
		return this->next().forward_propagation(y);
	}

	void update_weights() {
		b += b_gradientStorage * lr;
		this->next().update_weights();
	}

	void clear_stored_delta_gradients() {
		b_gradientStorage = 0;
		this->next().clear_stored_delta_gradients();
	}

	void set_batch_size(int x) {
		this->next().set_batch_size(x);
	}

	void write(std::ofstream& os) {
		w.write(os);
		this->next().write(os);
	}
	void read(std::ifstream& is) {
		w.read(is);
		this->next().read(is);
	}
	void init_storages() {
		w_gradientStorage = mat(this->OUTPUTS, this->INPUTS);
		w_gradientStorage = 0;
	}

};
}
}


#endif /* ADDITION_LAYER_H_ */
