#ifndef BlackCat_feedforward_h
#define BlackCat_feedforward_h
#include "Layer.h"
#include "BC_NeuralNetwork_Definitions.h"

class FeedForward: public Layer {

	nonLinearityFunction g;

	mat w_gradientStorage;
	vec b_gradientStorage;

	mat w;
	vec b;

public:
	FeedForward(unsigned inputs, unsigned outputs) {

		w = mat(outputs, inputs);
		b = vec(outputs);
		w_gradientStorage = mat(outputs, inputs);
		b_gradientStorage = vec(outputs);

		w.randomize(-3, 4);
		b.randomize(-2, 2);
	}
	~FeedForward() {}
	//NeuralNetwork algorithms
	vec forwardPropagation(vec x) {
		vec y = g(w * x + b);

		bpX.store(x);
		return next ? next->forwardPropagation(y) : y;
	}
	vec forwardPropagation_express(vec x) {
		vec y = g(w * x + b);

		return next ? next->forwardPropagation_express(y) : y;
	}
	vec backwardPropagation(vec dy) {
		vec x = bpX.poll_last();

		w_gradientStorage -=  dy * x.T(); //outer product
		b_gradientStorage -= dy;

		return prev ? prev->backwardPropagation(   w.T() * dy & g.d(x)    ) : dy;
	}
	vec backwardPropagation_ThroughTime(vec dy) {
		vec x = bpX.poll_last();

		w_gradientStorage -=  dy * x.T(); //outer product
		b_gradientStorage -= dy;
		return prev ? prev->backwardPropagation(   w.T() * dy & g.d(x)    ) : dy;
	}
	//NeuralNetwork update-methods
	void clearBackPropagationStorage() {
		w_gradientStorage = 0;
		b_gradientStorage = 0;
	}
	void clearGradientStorage() {
		bpX.clear();
	}
	void updateGradients() {
		w += w_gradientStorage & lr;
		b += b_gradientStorage & lr;
	}

};

#endif
