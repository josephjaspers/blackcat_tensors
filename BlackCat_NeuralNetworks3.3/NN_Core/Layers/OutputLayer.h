/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "Layer.h"

namespace BC {
namespace NN {

template<class derived>
struct OutputLayer : Layer<derived> {

	mat& y = this->prev().y;
	vec zero = vec(this->OUTPUTS);

public:

	OutputLayer(int inputs) : Layer<derived>(inputs) {
		zero.zero();
	}

	template<class expr> auto forward_propagation(const f_mat<expr>& x) {
		return x;
	}
	template<class expr> auto forward_propagation_express(const f_mat<expr>& x) const {
		return x;
	}
	template<class expr> auto back_propagation(const f_mat<expr>& exp) {
		return this->prev().back_propagation(y - exp);
	}
	template<class expr> auto back_propagation_throughtime() {
		return this->prev().back_propagation(zero);
	}


	void set_batch_size(int) {}
	void update_weights() {}
	void clear_stored_delta_gradients() {}
	void write(std::ofstream& is) {
	}
	void read(std::ifstream& os) {
	}
	void setLearningRate(fp_type learning_rate) {
		return;
	}

};

}
}



#endif /* FEEDFORWARD_CU_ */
