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
#include "Layers/InputLayer.h"
#include "Layers/OutputLayer.h"
//#include "LayerLinker.h"
namespace BC {
namespace NN {

template<class... layers>
struct NeuralNetwork {

	Chain<InputLayer, layers..., OutputLayer> network;

	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	template<class tensor> auto forward_propagation	(const tensor& x) { return network.fp(x); }
	template<class tensor> auto back_propagation	(const tensor& y) { return network.backprop(y); }


	void set_batch_size(int size) { network.set_batch_size(size); }// initlayer_inputs(); }
	void write(std::ofstream& os) { network.write(os); }
	void read(std::ifstream& is)  { network.read(is);  }

	void update_weights()  						{ network.update_weights(); }
	void clear_stored_delta_gradients() 		{ network.clear_stored_delta_gradients(); }
	void setLearningRate(fp_type learning_rate) { network.setLearningRate(learning_rate); }
};

}
}

#endif /* NEURALNETWORK_H_ */
