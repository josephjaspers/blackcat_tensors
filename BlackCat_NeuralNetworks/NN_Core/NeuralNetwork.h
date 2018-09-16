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

template<class... integers>
int sum(int x, integers... ints) {
	return x + sum(ints...);
}
int sum(int x) { return x; }

template<template<class> class... layers>
struct NeuralNetwork {

	Chain<InputLayer, layers..., OutputLayer> network;

	int batch_size = 1;
//	Linker<InputLayer, layers..., OutputLayer> network;

	template<class... integers>
	NeuralNetwork(integers... architecture) : network(architecture...) {}

	template<class tensor> auto forward_propagation				(const tensor& x) { return network.head().forward_propagation(x); }
	template<class tensor> auto forward_propagation_express		(const tensor& x) { return network.head().forward_propagation(x); }
	template<class tensor> auto back_propagation				(const tensor& y) { return network.tail().back_propagation(y); }
	template<class tensor> auto back_propagation_throughtime() 					  { return network.tail().back_propagation_throughtime(); }


	void set_batch_size(int size) { batch_size = size; network.head().set_batch_size(size); initlayer_inputs(); }
	void write(std::ofstream& os) { network.head().write(os); }
	void read(std::ifstream& is)  { network.head().read(is);  }

	void update_weights()  { network.head().update_weights(); }
	void clear_stored_delta_gradients() { network.head().clear_stored_delta_gradients(); }
	void setLearningRate(fp_type learning_rate) { network.head().setLearningRate(learning_rate); }

	void initlayer_inputs() {
		int input_base_index = 0;
		auto& ws = forward_propagation_memories.get_workspace();
		network.head().init_input_view(ws, input_base_index, batch_size);
	}
	struct {

		int ws_size() {
			return current_timestamp.size();
		}

		vec& get_workspace() {
			return current_timestamp;
		}

		vec current_timestamp;
		std::forward_list<vec> previous_timestamps;

		void resize(int new_sz) {
			if (current_timestamp.size() != new_sz)
				current_timestamp = vec(new_sz);
		}
		void push() {
			int workspace_sz = current_timestamp.size();
			previous_timestamps.push_front(std::move(current_timestamp));
			current_timestamp = vec(workspace_sz);
		}
		void pop() {
			previous_timestamps.pop_front();
		}
	} forward_propagation_memories;
};

}
}

#endif /* NEURALNETWORK_H_ */
