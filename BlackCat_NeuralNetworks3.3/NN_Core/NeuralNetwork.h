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

	template<class tensor> auto forwardPropagation				(const tensor& x) { return network.head().forwardPropagation(x); }
	template<class tensor> auto forwardPropagation_Express		(const tensor& x) { return network.head().forwardPropagation(x); }
	template<class tensor> auto backPropagation					(const tensor& y) { return network.tail().backPropagation(y); }
	template<class tensor> auto backPropagation_throughtime() 					  { return network.tail().backPropagation_throughtime(); }

	void write(std::ofstream& os) { network.head().write(os); }
	void read(std::ifstream& is)  { network.head().read(is);  }

	auto updateWeights()  { return network.head().updateWeights(); }
	auto clearBPStorage() { return network.head().clearBPStorage(); }

	void setLearningRate(fp_type learning_rate) { network.head().setLearningRate(learning_rate); }
};

}
}

#endif /* NEURALNETWORK_H_ */
