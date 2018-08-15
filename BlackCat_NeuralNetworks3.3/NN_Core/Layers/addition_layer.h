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


#include "Layer.h"
#include <mutex>

namespace BC {
namespace NN {

template<class derived>
struct Addition : public Layer<derived> {
public:

	using Layer<derived>::lr;
	vec w_gradientStorage;

	vec w;
	mat& x = this->prev().y;

	Addition(int inputs) :
			Layer<derived>(inputs),
			w(this->INPUTS),
			w_gradientStorage(this->INPUTS)
	{
		w.randomize(-1, 1);
		init_storages();
	}

	template<class e> auto forward_propagation(const expr::mat<e>& x) {
		return this->next().forward_propagation(w + x);
	}
	template<class e> auto back_propagation(const expr::mat<e>& dy_) {
		vec dy = dy_;
		w_gradientStorage -= dy;
		return this->prev().back_propagation(dy_);
	}
	template<class e> auto forward_propagation_express(const expr::mat<e>& x) const {
		auto y = x + w;
		return this->next().forward_propagation(y);
	}

	void update_weights() {
		w += w_gradientStorage * lr;
		this->next().update_weights();
	}

	void clear_stored_delta_gradients() {
		w_gradientStorage = 0;
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
