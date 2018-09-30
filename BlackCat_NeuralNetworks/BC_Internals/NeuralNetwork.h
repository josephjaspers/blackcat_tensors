/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *      Author: joseph
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <forward_list>

#include "../BC_Internals/LayerChain.h"
#include "../BC_Internals/Layers/InputLayer.h"
#include "../BC_Internals/Layers/OutputLayer.h"
#include "../BC_Internals/Layers/Recurrent_InputLayer.h"
#include "../BC_Internals/Layers/Recurrent_OutputLayer.h"

//#include "LayerLinker.h"
namespace BC {
namespace NN {

template<class... Ts>
struct recurrent_determiner {
	static constexpr bool value = false;
};
template<class T, class... Ts>
struct recurrent_determiner<T, Ts...> {
	static constexpr bool value = T::is_recurrent || recurrent_determiner<Ts...>::value;
};

template<class... layers>
struct NeuralNetwork {


	static constexpr bool is_recurrent = recurrent_determiner<layers...>::value;
	using input_t = std::conditional_t<is_recurrent, Recurrent_InputLayer, InputLayer>;
	using output_t = std::conditional_t<is_recurrent, Recurrent_OutputLayer, OutputLayer>;

	Chain<input_t, layers..., output_t> network;
	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	template<class tensor> auto forward_propagation	(const tensor& x) { return network.fp(x); }
	template<class tensor> auto back_propagation	(const tensor& y) { return network.back_propagation(y); }
	void update_weights()  { network.update_weights(); }
	void cache_gradients() { network.cache_gradients(); }
	void set_learning_rate(fp_type lr)  { network.set_learning_rate(lr);  }
	void set_max_bptt_length(int len)   { network.set_max_bptt_length(len); }
	void set_batch_size(int size) 		{ network.set_batch_size(size);   }
	void initialize_variables()      	{ network.initialize_variables(); }

	void write(std::ofstream& os) { network.write(os); }
	void read(std::ifstream& is)  { network.read(is);  }

};
}
}

#endif /* NEURALNETWORK_H_ */
