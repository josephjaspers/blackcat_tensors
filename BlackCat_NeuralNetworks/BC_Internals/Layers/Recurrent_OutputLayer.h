/*
 * Recurrent_OutputLayer.h
 *
 *  Created on: Sep 27, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_LAYERS_RECURRENT_OUTPUTLAYER_H_
#define BC_INTERNALS_LAYERS_RECURRENT_OUTPUTLAYER_H_

#include <vector>

#include "Recurrent_Layer_Base.h"
namespace BC {
namespace NN{

struct Recurrent_OutputLayer : Recurrent_Layer_Base {

	using Recurrent_Layer_Base::t;

	std::vector<mat_view> x =  std::vector<mat_view>(this->get_max_bptt_length());

public:

	Recurrent_OutputLayer(int inputs) : Recurrent_Layer_Base(inputs, inputs) {}

	template <class T>
	const auto& forward_propagation(const expr::mat<T>& x_) {
		 x[t] = mat_view(x_);
		 t++;
		 return x[t];
	}
	template<class T>
	auto back_propagation(const expr::mat<T>& exp) {
		return x[t] - exp;
	}


	void update_weights() {}
	void cache_gradients() { t--; }
	void clear_stored_gradients() {}
	void write(std::ofstream& is) {}
	void read(std::ifstream& os) {}
	void set_learning_rate(fp_type learning_rate) {}

	void set_max_bptt_length(int length) {
		Recurrent_Layer_Base::set_max_bptt_length(length);
		x = std::vector<mat_view>(length);
	}

	auto& outputs() { return x; }
	auto& inputs()  { return x; }
	auto& deltas()  { return NULL_TENSOR; }
	auto& weights()	{ return NULL_TENSOR; }
	auto& bias()	{ return NULL_TENSOR; }
};

}
}


#endif /* BC_INTERNALS_LAYERS_RECURRENT_OUTPUTLAYER_H_ */
