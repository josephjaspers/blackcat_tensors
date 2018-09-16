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

	vec zero = vec(this->OUTPUTS);

	auto& x() { return this->prev().y; }

public:

	OutputLayer(int inputs) : Layer_Base<derived>(inputs) {
		zero.zero();
	}

	template<class t> mat forward_propagation(const expr::mat<t>& x) {
		return x;
	}
	template<class t> auto forward_propagation_express(const expr::mat<t>& x) {
		return x;
	}
	template<class t> auto back_propagation(const expr::mat<t>& exp) {
		return this->prev().back_propagation(x() - exp);
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


	template<class function>
	void for_each(function f) {}
	void init_input_view(vec& workspace, int& offset, int batch_size) {	}
};

}
}



#endif /* FEEDFORWARD_CU_ */
