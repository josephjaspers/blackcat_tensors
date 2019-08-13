/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_
#define BLACKCAT_NEURALNETWORK_NEURALNETWORK_H_

#include "Layers/Layer_Traits.h"
#include "Layer_Chain.h"

#include <sys/types.h>
#include <sys/stat.h>

namespace BC {
namespace nn {
namespace detail {

template<class T>
using is_recurrent_layer = BC::traits::conditional_detected_t<
			detail::query_forward_requires_outputs, T, std::false_type>;


}


//HEAD
template<class... Layers>
struct NeuralNetwork {

	static constexpr bool recurrent_neural_network =
			BC::traits::any<detail::is_recurrent_layer, Layers...>::value;

    using self = NeuralNetwork<Layers...>;
    using layer_chain = LayerChain<recurrent_neural_network, 0, void, Layers...>;

    layer_chain m_layer_chain;

    NeuralNetwork(Layers... layers):
    	m_layer_chain(layers...) {}

	template<class T> auto back_propagation(const T& tensor_expected) {
		return m_layer_chain.tail().bp(tensor_expected);
	}

	template<class T> auto forward_propagation(const T& tensor_expected) {
		return m_layer_chain.head().fp(tensor_expected);
	}

	template<int X> auto& get_layer() const { return m_layer_chain.get(BC::traits::Integer<X>()); }
	template<int X> auto& get_layer() { return m_layer_chain.get(BC::traits::Integer<X>()); }

	void set_learning_rate(double lr) { m_layer_chain.for_each([&](auto& layer) { layer.set_learning_rate(lr); }); }
    void set_batch_size(int x) { m_layer_chain.for_each([&](auto& layer) { layer.set_batch_size(x);    });}
    void update_weights()      { m_layer_chain.for_each([ ](auto& layer) { layer.update_weights();        });}


    /**
     * Returns a yaml representation of the neural network
     */
    std::string get_string_architecture() const {
    	std::string architecture = "";
    	m_layer_chain.for_each([&](auto& layer) {
    		architecture += layer.get_string_architecture() + "\n";
    	});
    	return architecture;
    }

    /**
     * Creates the directory `directory_name` using mkdir.
     * Than outputs an architecture.yaml file with a description of the neural network.
     * Than iterates through each layer and calls 'save(int layer_index, std::string directory_name)'
     */
    void save(std::string directory_name) {
    	//Attempt to create a directory to save our model in
    	if ((directory_name != "") && (directory_name != ".")) {
    		int error = system(std::string("mkdir " + directory_name).c_str());
    	}

		//Create a yaml file with network description/architecture
		std::string architecture_yaml = directory_name + "/architecture.yaml";
		std::ofstream os(architecture_yaml);
		os << get_string_architecture();

		//Iterate through each layer and call save
    	int index = 0;
    	m_layer_chain.for_each([&](auto& layer){
    		layer.save(index, directory_name);
    		index++;
    	});
    }

    void load(std::string directory_name) {
    	int index = 0;
    	m_layer_chain.for_each([&](auto& layer){
    		layer.load(index, directory_name);
    		index++;
    	});
    }
};

template<class... Layers>
auto neuralnetwork(Layers... layers) {
	return NeuralNetwork<Layers...>(layers...);
}

}
}

#endif /* NEURALNETWORK_H_ */
