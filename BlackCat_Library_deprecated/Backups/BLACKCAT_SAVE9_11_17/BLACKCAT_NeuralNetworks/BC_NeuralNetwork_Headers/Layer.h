/*
 * Layer.h
 *
 *  Created on: Aug 10, 2017
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_
#include "NeuralNetwork_Structs.h"
#include "BC_NeuralNetwork_Definitions.h"
#include <ostream>
#include <istream>
#include <mutex>
#include "unq_thread.h"


class Layer {
public:
	nonLinearityFunction g;			//sigmoid
	nonLinearityFunction h;			//hyperbolic tangent

	scalar lr = scalar(.3);			//learning rate

	Layer* next = nullptr;			//next layer in NN
	Layer* prev = nullptr;			//prev layer in NN

	bpStorage bpX;
	bpStorage bpY;

	const tensor& xt() { return bpX(); }
	const tensor& yt() { return bpY(); }

public:
	//Constructors
	virtual ~Layer() {}
	void link(Layer* l) { next = l; l->prev = this; }

	//NeuralNetwork algorithms
	virtual vec forwardPropagation(vec x) = 0;
	virtual vec forwardPropagation_express(vec x) = 0;
	virtual vec backwardPropagation(vec dy) = 0;
	virtual vec backwardPropagation_ThroughTime(vec dy) = 0;

	//NeuralNetwork update-methods
	virtual void clearBackPropagationStorage() = 0;
	virtual void clearGradientStorage() = 0;
	virtual void updateGradients() = 0;

	//Accessors/Mutators
	double getLearningRate() { return lr(); }
	void setLearningRate(double lr) {this->lr = lr;}


	//read/write
	// write(std::ofstream& os);
	//void read(std::ifstream& is);
};

#endif /* LAYER_H_ */
