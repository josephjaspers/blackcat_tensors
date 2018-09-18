/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUT_CU
#define OUTPUT_CU

#include "Layer_Base.h"

namespace BC {
namespace NN {

//template<class derived>
struct InputLayer : Layer_Base {

	InputLayer(int inputs, int outputs) : Layer_Base(inputs, outputs) {}

	mat_view y;

	template<class tensor> const auto& forward_propagation(const tensor& x) {
		return y = mat_view(x);
	}


	template<class tensor> auto back_propagation(const tensor& dy) {
		return dy;
	}

	void set_batch_size(int x) {}

	void update_weights() {}
	void clear_stored_delta_gradients() {}
	void write(std::ofstream& os) {}
	void read(std::ifstream& is) {}
	void setLearningRate(fp_type learning_rate) {}
};
}
}
#endif /* FEEDFORWARD_CU_ */
