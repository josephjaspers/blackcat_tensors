/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "Layer_Base.h"

namespace BC {
namespace NN {

template<class derived>
struct OutputLayer : Layer_Base<derived> {

	const mat& x_() const { return this->prev().y; }

	vec zero = vec(this->OUTPUTS);

public:

	OutputLayer(int inputs) : Layer_Base<derived>(inputs) {
		zero.zero();
	}

	template<class t> const auto& forward_propagation(const expr::mat<t>& x) {
		return x;
	}
	template<class t> auto forward_propagation_express(const expr::mat<t>& x) {
		return x;
	}
	template<class t> auto back_propagation(const expr::mat<t>& exp) {
		return x_() - exp;
//		return this->prev().back_propagation(x_() - exp);
	}
	 auto back_propagation_throughtime() {
		return this->prev().back_propagation(zero);
	}


	void set_batch_size(int) {}
	void update_weights() {}
	void clear_stored_delta_gradients() {}
	void write(std::ofstream& is) {}
	void read(std::ifstream& os) {}
	void setLearningRate(fp_type learning_rate) {}

};

}
}



#endif /* FEEDFORWARD_CU_ */
