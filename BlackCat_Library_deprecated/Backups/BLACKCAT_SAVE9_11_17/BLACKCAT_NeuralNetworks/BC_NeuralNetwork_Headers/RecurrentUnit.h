/*
 * RecurrentLayer.h
 *
 *  Created on: Aug 21, 2017
 *      Author: joseph
 */

#ifndef RECURRENTLAYER_H_
#define RECURRENTLAYER_H_

class RecurrentUnit : public Layer {
	nonLinearityFunction g;


	vec c;

	mat w;
	mat r;
	vec b;

	vec dc;

	mat w_gradientStorage;
	mat r_gradientStorage;
	vec b_gradientStorage;

public:
	RecurrentUnit(unsigned inputs, unsigned outputs) {

		c = vec(outputs);
		dc = vec(outputs);

		w = mat(outputs, inputs);
		r = mat(outputs, outputs);

		b = vec(outputs);

		w_gradientStorage = mat(outputs, inputs);
		r_gradientStorage = mat(outputs, outputs);
		b_gradientStorage = vec(outputs);

		r.randomize(-3, 0);
		b.randomize(-3, 4);
		w.randomize(-3, 4);
	}
	virtual ~RecurrentUnit() {

	}
	vec forwardPropagation(vec x)  override {
		c = g(w * x + r * c + b);

		bpX.store(x);
		bpY.store(c);
		return next ? next->forwardPropagation(c) : c;
	}
	vec forwardPropagation_express( vec x) override {
		c = g(w * x + r * c + b);

		return next ? next->forwardPropagation_express(c) : c;
	}
	vec backwardPropagation(vec dy) override {
		vec xt = bpX.poll_last();
		vec yt = bpY.poll_last();


		w_gradientStorage -= dy * xt.T();
		b_gradientStorage -= dy;
		r_gradientStorage -= dc * yt.T();


		dc = dy;
		vec dx = (w.T() * dy) & g.d(xt);

		if (prev) {
			return prev->backwardPropagation(dx);
		} else {
			return dy;
		}
	}
	vec backwardPropagation_ThroughTime(vec dy) override {
		vec xt = bpX.poll_last();
		vec yt = bpY.poll_last();


		w_gradientStorage -= dy * xt.T();
		b_gradientStorage -= dy;
		r_gradientStorage -= dc * yt.T();


		dc += dy + (r.T() * dc) & g.d(yt);
		vec dx = (w.T() * dy) & g.d(xt);

		if (prev) {
			return prev->backwardPropagation_ThroughTime(dx);
		} else {
			return dy;
		}
	}


	void clearBackPropagationStorage() {
		bpX.clear();
	}
	void clearGradientStorage() {
		dc = 0;

		w_gradientStorage = 0;
		r_gradientStorage = 0;
		b_gradientStorage = 0;

	}
	void updateGradients() {
		w += w_gradientStorage & lr;
		r += r_gradientStorage & lr;
		b += b_gradientStorage & lr;
	}
};
#endif /* RECURRENTLAYER_H_ */
