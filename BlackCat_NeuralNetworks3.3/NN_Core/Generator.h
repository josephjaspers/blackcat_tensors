/*
 * Generator.h
 *
 *  Created on: Feb 22, 2018
 *      Author: joseph
 */

#ifndef GENERATOR_H_
#define GENERATOR_H_

#include "NeuralNetwork.h"

namespace BC {

template<template<class> class... layers, class... integers>
auto generateNetwork(integers... structure) {
	return NeuralNetwork<layers...>(structure...);
}



}



#endif /* GENERATOR_H_ */
