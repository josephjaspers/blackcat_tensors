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

template<template<class> class... layers>
struct NeuralNetwork {

	LayerChain<BASE, InputLayer, layers..., OutputLayer> network;

	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	auto forwardPropagation(const vec& x) { return network.head().forwardPropagation(x); }
	auto forwardPropagation_Express(const vec& x) { return network.head().forwardPropagation(x); }
	auto backPropagation(const vec& y) { return network.tail().backPropagation(y); }
	auto train(const vec& x, const vec& y) { return network.head().train(x, y); }

	auto updateWeights() { return network.head().updateWeights(); }
	auto clearBPStorage() { return network.head().clearBPStorage(); }
	void setLearningRate(fp_type learning_rate) { network.head().setLearningRate(learning_rate); }


};





}

#endif /* NEURALNETWORK_H_ */
