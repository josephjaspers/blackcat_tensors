#pragma once
#ifndef Layer_h
#define Layer_h
#include "stdafx.h"
#include "Matrices.h"
#include "nonLinearityFunction.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace Matrices;
	/*
		Layers are designed as nodes in a double-linked list. Calling methods such as forward propagation call the next layer forward propagation
		in a chain. Same is true for backward propagation, deconstruction, saving and writing. The NeuralNetwork class is the "list head" 
		and stores the pointers input and output which may be reinterperted to first and last. 

		Similarily Neural Network extends Layer, and as such maybe used interchangeabely as its own layer. 
	*/


class Layer {
	//super class of all sub class layers
protected:
	nonLinearityFunct g;								//An object that applies non linearity functions and derivatives to apropriate Vectors

public:
	const Vector INPUT_ZEROS;							//An all 0's Vector of size n = numb_inputs
	const Vector OUTPUT_ZEROS;							//An all 0's Vector of size n = numb_outputs
	const int NUMB_INPUTS;								//number of inputs
	const int NUMB_OUTPUTS;								//number of outputs

	virtual ~Layer() {
		if (next != nullptr) {
			delete next;
		}
	}

protected:
														//superclass constructor -- initializes constants
	Layer(int inputs, int outputs) : INPUT_ZEROS(inputs), OUTPUT_ZEROS(outputs), NUMB_INPUTS(inputs), NUMB_OUTPUTS(outputs) {

	}				
	typedef std::vector<Vector> bpStorage;				//type defination -- this format is used consistently to store the activations for backprop through time 
	Layer* next = nullptr;										//pointer to next layer in network --similair to LinkedList 
	Layer* prev = nullptr;										//pointer to previous layer in network --similair to LinkedList
	double lr = .3;										//Learning Rate
	double mr = .01;									//Momentum Rate (currently not supported)
	
public:
													//Method for linking two layers together
	void link(Layer& l) {
		next = &l; //next = linked layer 
		l.prev = this; //next layers "prev" is set to this 
	}

	double getLearningRate() { return lr; }				//accessors for lr/mr
	double getMomentumRate() { return mr; }
	void setLearningRate_link(double lr) {
		Layer::lr = lr; if (next != nullptr) { next->setLearningRate_link(lr); }
	}
	void setMomentumRate_link(double mr) {
		Layer::mr = mr; if (next != nullptr) { next->setMomentumRate_link(mr); }
	}
	void setLearningRate(double lr) { Layer::lr = lr; }	//mutators for lr/mr
	void setMomentumRate(double mr) { Layer::mr = mr; }
	int getInputs() { return NUMB_INPUTS; }				//accessors for inputs/outputs
	int getOutputs() { return NUMB_OUTPUTS; }
	
	//should be chaind 
//	static void read(std::ifstream& is);  //pseudo abstract method. Each layer should have a static read method 
	
	virtual void write_list(std::ofstream& os) { writeClass(os); write(os); if (next) next->write_list(os); } //write the class, write the layer, do same for next layer
	virtual void write(std::ofstream& os) = 0; //Only the weights are saved  -- does not save activations
	virtual void writeClass(std::ofstream& os) = 0; //writes the classname --> accessed so collection classes may read apropriate NN (IE Neural Network)
public:

	//Chained Methods
	virtual Vector forwardPropagation_express(const Vector& x) = 0;						//forward propagation [Express does not store activations for BPPT]
	virtual Vector forwardPropagation(const Vector& x) = 0;								//forward propagation [Stores activations for BP & BPPT]
	virtual Vector backwardPropagation(const Vector& dy) = 0;							//backward propagation[Initial BP] 
	virtual Vector backwardPropagation_ThroughTime(const Vector& dy) = 0;				//BPPT [Regular BP must be called before BPTT]
	//Chained Methods											
	virtual void clearBPStorage() { if (next) next->clearBPStorage(); }				//clears the stored activations used in BP of the layers --> Bpstorage will accumulate after a while 
	virtual void clearGradients() { if (next) next->clearGradients(); }					//clears the stored gradients of the past back propagation --> Clears the store gradients 
	virtual void updateGradients() { if (next) next->updateGradients(); }			//updates the weights using the stored gradients. 

	Layer& setSigmoid() { g.setSigmoid(); }				//set the crush function of the layer to sigmoid (default = sigmoid)
	Layer& setTanh() { g.setTanh(); }					//set the crush function of the layer to tanh
	Layer& setNonLinearity(int i) { g.setNonLinearityFunction(i); } //set the nonlinearity to an a function --designed to allow easy updates to nonlinearity functions || Currently: 0 = sigmoid, 1 = tanh, 2 = softmax

};
#endif
