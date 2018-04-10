#ifndef BLACKCAT_neuralnet_h
#define  BLACKCAT_neuralnet_h

#include "Layer.h"
#include <cstdlib>
class NeuralNetwork : public Layer {

	Layer* last = nullptr;
	Layer* first = nullptr;
public:

	void add(Layer* l) {
		if (first == nullptr) {
			first = l;
			last = first;
		} else {
			last->next = l;
			l->prev = last;
			last = l;
		}
	}
	vec forwardPropagation(vec x) {
		return first->forwardPropagation(x);
	}
	vec forwardPropagation_express(vec x) {
		return first->forwardPropagation_express(x);
	}
	vec backwardPropagation(vec x) {
		return last->backwardPropagation(x);
	}
	vec backwardPropagation_ThroughTime(vec x) {
		return last->backwardPropagation_ThroughTime(x);
	}

	void clearBackPropagationStorage() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->clearBackPropagationStorage();
			ref_first = ref_first->next;
		}
	}
	void clearGradientStorage() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->clearGradientStorage();
			ref_first = ref_first->next;
		}
	}
	void updateGradients() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->updateGradients();
			ref_first = ref_first->next;
		}
	}
	void setLearningRate(double lr) {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->setLearningRate(lr);
			ref_first = ref_first->next;
		}
	}
	void train(const std::vector<vec>& inputs, const std::vector<vec>& outputs, unsigned iters) {
		while (iters > 0) {
			train(inputs, outputs);
			--iters;
		}
	}

	void update() {
		Layer* ref_first = first;
		while (ref_first != last->next) {
			ref_first->updateGradients();
			ref_first->clearGradientStorage();
			ref_first->clearBackPropagationStorage();

			ref_first = ref_first->next;
		}
	}

	void train(const std::vector<vec>& inputs, const std::vector<vec>& outputs) {

		for (unsigned i = 0; i < inputs.size(); ++i) {
			vec hyp = forwardPropagation(inputs[i]);
			vec res = hyp - outputs[i];
			backwardPropagation(res);
			update();
		}

	}

};

#endif
