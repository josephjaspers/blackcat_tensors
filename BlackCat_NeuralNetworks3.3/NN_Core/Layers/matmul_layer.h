/*
 * matmul_layer.h
 *
 *  Created on: Jul 29, 2018
 *      Author: joseph
 */

#ifndef MATMUL_LAYER_H_
#define MATMUL_LAYER_H_

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
struct matmul_layer : public Layer<derived> {
public:

	using Layer<derived>::lr;					//the learning rate

	mat w_gradientStorage;		//weight gradient storage

	mat y;							//outputs

	mat w;							//weights
	mat& x = this->prev().y;

	matmul_layer(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, inputs),

			w_gradientStorage(this->OUTPUTS, this->INPUTS)
	{
		w.randomize(-2, 2);
		init_storages();
	}

	template<class e> auto forward_propagation(const expr::mat<e>& x) {
		y = w * x;

		return this->next().forward_propagation(y);
	}
	template<class e> auto back_propagation(const expr::mat<e>& dy_) {
		mat dy = dy_; //cache the values (avoid recomputing dy_)

		w_gradientStorage -= dy * x.t();

		return this->prev().back_propagation(w.t() * dy);
	}
	template<class e> auto forward_propagation_express(const expr::mat<e>& x) const {
		return this->next().forward_propagation_express(w * x);
	}

	void update_weights() {
		w += w_gradientStorage * lr;

		this->next().update_weights();
	}

	void clear_stored_delta_gradients() {
		w_gradientStorage = 0;	//gradient lists

		this->next().clear_stored_delta_gradients();
	}

	void set_batch_size(int x) {
		y = mat(this->OUTPUTS, x);
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


#endif /* MATMUL_LAYER_H_ */
