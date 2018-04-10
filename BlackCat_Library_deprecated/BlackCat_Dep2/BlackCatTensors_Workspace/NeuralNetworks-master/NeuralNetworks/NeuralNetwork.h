#pragma once
#ifndef NeuralNetwork_h
#define NeuralNetwork_h
#include "stdafx.h"
//superclass layer.h
#include "Layer.h"
//include all the layers
#include "FeedForward.h"
#include "FF_norec.h"
#include "RecurrentUnit.h"
#include "GRU.h"
#include "LSTM.h"

class NeuralNetwork : public Layer {

	Layer* input;	//first of linked list
	Layer* output;	//last of linked list

	int size;

	bpStorage bpO;
	const Vector& Ot() { return bpO.back(); }

public:
	void push_back(Layer* l);

	NeuralNetwork();
	~NeuralNetwork();

	void setLearningRate(double lr) {
		if (input != nullptr) {
			input->setLearningRate_link(lr);
		}
	}

	Vector forwardPropagation_express(const Vector& input);
	Vector forwardPropagation(const Vector& input);
	Vector backwardPropagation(const Vector& y);
	Vector backwardPropagation_ThroughTime(const Vector& dy);

	void train(std::vector<Vector>& x, Vector& y);
	void train(Vector x, Vector y);
	void train(std::vector<Vector>& x, std::vector<Vector>& y);
	void train(std::vector<std::vector<double>> x, std::vector<double> y);
	void train(std::vector<double> x, std::vector<double> y);

	Vector NeuralNetwork::predict(std::vector<Vector> x) {
		for (int i = 0; i < x.size() - 1; ++i) {
			forwardPropagation_express(x[i]);
		}
		return forwardPropagation_express(x.back());
	}
	Vector NeuralNetwork::predict(Vector x) {
		return forwardPropagation_express(x);
	}
	Vector NeuralNetwork::predict(std::vector<std::vector<double>> x) {
		for (int i = 0; i < x.size() - 1; ++i) {
			forwardPropagation_express(Vector(x[i]));
		}
		return forwardPropagation_express(Vector(x.back()));

	}

	void clearBPStorage();
	void clearGradients();
	void updateGradients();

	void write(std::ofstream& os);
	void writeClass(std::ofstream& os);
	NeuralNetwork& read(std::ifstream& is);

private:
	static Layer* readLayer(std::ifstream& is);
};
#endif
