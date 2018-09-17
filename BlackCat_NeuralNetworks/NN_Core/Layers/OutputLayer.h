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

	mat_view x;

public:

	OutputLayer(int inputs) : Layer_Base<derived>(inputs, inputs) {}

	template<class t> const auto& forward_propagation(const expr::mat<t>& x_) {
		return x = mat_view(x_);
	}
	template<class t> auto forward_propagation_express(const expr::mat<t>& x_) {
		return x = mat_view(x_);
	}
	template<class t> auto back_propagation(const expr::mat<t>& exp) {
		return x - exp;
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
