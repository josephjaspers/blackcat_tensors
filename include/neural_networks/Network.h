/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *      Author: joseph
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <forward_list>
#include "LayerChain.h"

namespace BC {
namespace nn {

template<class... layers>
struct NeuralNetwork {

    Chain<layers...> network;
    template<class... integers>
    NeuralNetwork(integers... architecture) : network(architecture...) {}

	template<class tensor> auto& forward_propagation(const tensor& x) {
		return network.forward_propagation(x);
	}
	template<class tensor> auto back_propagation(const tensor& y) {
		return network.back_propagation(y);
	}

    void update_weights()  { network.update_weights(); }
    void cache_gradients() { network.cache_gradients(); }
    void set_max_bptt_length(int len)   { network.set_max_bptt_length(len); }
    void set_batch_size(int size)       { network.set_batch_size(size);   }

};

template<class... Layers>
auto neuralnetwork(Layers&&... layers) {
	return NeuralNetwork<Layers...>(layers...);
}

}
}

#endif /* NEURALNETWORK_H_ */
