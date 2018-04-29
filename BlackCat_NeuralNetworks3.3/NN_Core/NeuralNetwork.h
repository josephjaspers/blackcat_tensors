/*
 * NeuralNetwork.h
 *
 *  Created on: Mar 5, 2018
 *      Author: joseph
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "LayerChain.h"
#include "Defaults.h"
namespace BC {
namespace NN {
template<template<class> class... layers>
struct NeuralNetwork {

	LayerChain<BASE, InputLayer, layers..., OutputLayer> network;

	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	auto forwardPropagation(const vec& x) { return network.head().forwardPropagation(x); }
	auto forwardPropagation_Express(const vec& x) { return network.head().forwardPropagation(x); }
	auto backPropagation(const vec& y) { return network.tail().backPropagation(y); }
	auto backPropagation_throughtime() { return network.tail().backPropagation_throughtime(); }

	void write(std::ofstream& os) { network.head().write(os); }
	void read(std::ifstream& is) { network.head().read(is); }

	auto train(const vec& x, const vec& y) { return network.head().train(x, y); }

	auto updateWeights() { return network.head().updateWeights(); }
	auto clearBPStorage() { return network.head().clearBPStorage(); }
	void setLearningRate(fp_type learning_rate) { network.head().setLearningRate(learning_rate); }
	void set_omp_threads(int i) { network.head().set_omp_threads(i); }


};

}
}

#endif /* NEURALNETWORK_H_ */
