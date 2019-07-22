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

//HEAD
template<class... Layers>
struct NeuralNetwork {

    using self = NeuralNetwork<Layers...>;
    using parent = LayerChain<0, void, Layers...>;

    LayerChain<0, void, Layers...> m_layer_chain;

    NeuralNetwork(Layers... layers):
    	m_layer_chain(layers...) {}

	template<class T> auto back_propagation(const T& tensor_expected) {
		return m_layer_chain.tail().bp(tensor_expected);
	}

	template<class T> auto forward_propagation(const T& tensor_expected) {
		return m_layer_chain.head().fp(tensor_expected);
	}

    void set_batch_size(int x) { m_layer_chain.for_each([&](auto& layer) { layer.set_batch_size(x);    });}
    void update_weights()      { m_layer_chain.for_each([ ](auto& layer) { layer.update_weights();        });}
//    void read(std::ifstream& is)  { m_layer_chain.for_each([&](auto& layer) { layer.read(is);     });}
//    void write(std::ifstream& os) { m_layer_chain.for_each([&](auto& layer) { layer.write(os);     });}
//    void set_max_bptt_length(int len) { m_layer_chain.for_each([&](auto& layer)  { layer.set_max_bptt_length(len);}); }
};
template<class... Layers>
auto neuralnetwork(Layers&&... layers) {
	return NeuralNetwork<Layers...>(layers...);
}

}
}

#endif /* NEURALNETWORK_H_ */
