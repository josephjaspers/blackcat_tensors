/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *      Author: joseph
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "LayerChain.h"
namespace BC {
namespace NN {
template<template<class> class... layers>
struct NeuralNetwork {

	LayerChain<BASE, InputLayer, layers..., OutputLayer> network;

	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	template<class tensor> auto forward_propagation				(const tensor& x) { return network.head().forward_propagation(x); }
	template<class tensor> auto forward_propagation_express		(const tensor& x) { return network.head().forward_propagation(x); }
	template<class tensor> auto back_propagation					(const tensor& y) { return network.tail().back_propagation(y); }
	template<class tensor> auto back_propagation_throughtime() 					  { return network.tail().back_propagation_throughtime(); }


	void set_batch_size(int size) { network.head().set_batch_size(size); }
	void write(std::ofstream& os) { network.head().write(os); }
	void read(std::ifstream& is)  { network.head().read(is);  }

	auto update_weights()  { return network.head().update_weights(); }
	auto clear_stored_delta_gradients() { return network.head().clear_stored_delta_gradients(); }

	void setLearningRate(fp_type learning_rate) { network.head().setLearningRate(learning_rate); }
};

}
}

#endif /* NEURALNETWORK_H_ */
