/*
 * Generator.h
 *
 *  Created on: Feb 22, 2018
 *      Author: joseph
 */

#ifndef GENERATOR_H_
#define GENERATOR_H_


#include "InputLayer.h"
#include "OutputLayer.cu"
#include "Defaults.h"


#include "LayerChain.cu"


namespace BC {

template<template<class> class... layers, class... integers>
auto generateNetwork(integers... structure) {
	auto net = LayerChain<BASE, InputLayer, layers..., OutputLayer>(structure...);
	return net;
}



}



#endif /* GENERATOR_H_ */
