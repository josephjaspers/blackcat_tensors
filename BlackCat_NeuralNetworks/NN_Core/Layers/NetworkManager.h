/*
 * NetworkManager.h
 *
 *  Created on: Sep 14, 2018
 *      Author: joseph
 */

#ifndef NN_CORE_LAYERS_NETWORKMANAGER_H_
#define NN_CORE_LAYERS_NETWORKMANAGER_H_

#include "LayerManager.h"

namespace BC {
namespace NN {

template<class... layers>
class NetworkManager {

	LayerManager<layers...> network;

	vec biases;
	vec weights;
	vec bias_gradients;
	vec weight_gradients;
	vec deltas;
	vec inputs;

	template<class... layer_params>
	NetworkManager(layer_params... params)
	: network(params...),
	  biases(network.bias_workspace_size()),
	  weights(network.weight_workspace_Size()),
	  deltas(network.delta_workspace_size()),
	  inputs(network.inputs_workspace_size())
	 {}
};
}
}

#endif /* NN_CORE_LAYERS_NETWORKMANAGER_H_ */
