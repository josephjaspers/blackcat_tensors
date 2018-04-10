/*
 * RecurrentLayer.h
 *
 *  Created on: Aug 21, 2017
 *      Author: joseph
 */

#ifndef Gated_red
#define Gated_red

class GatedRecurrentUnit : public Layer {
	nonLinearityFunction g;

	vec c;
	vec dc;

	vec x;

	vec i, f;
	vec di, df;

	mat wi, wf;
	mat ri, rf;
	vec bi, bf;

	bpStorage bpI, bpF;

	mat wi_gradientStorage, wf_gradientStorage;
	mat ri_gradientStorage, rf_gradientStorage;
	vec bi_gradientStorage, bf_gradientStorage;

public:
	GatedRecurrentUnit(unsigned inputs, unsigned outputs) {
		x = vec(inputs);
		c = vec(outputs);
		dc = vec(outputs);

		wi = wf = mat(outputs, inputs);
		ri = rf = mat(outputs, outputs);

		bi = bf = vec(outputs);

		wi_gradientStorage = wf_gradientStorage = mat(outputs, inputs);
		ri_gradientStorage = rf_gradientStorage = mat(outputs, outputs);
		bi_gradientStorage = bf_gradientStorage = vec(outputs);

		ri.randomize(-3, 2);
		bi.randomize(-3, 4);
		wi.randomize(-3, 4);

		//Initialize forget gates into negative range (train to remember)
		rf.randomize(-4, 0);
		bf.randomize(-4, 0);
		wf.randomize(-4, 0);
	}
	virtual ~GatedRecurrentUnit() {

	}
	vec forwardPropagation(vec a) override {

		bpX.store(x);
		bpI.store(i);
		bpF.store(f);
		bpY.store(c);

		x = a;
		i = g(wi * x + ri * c + bi);
		f = g(wf * x + rf * c + bf);

		c &= f;
		c += i;

		return next ? next->forwardPropagation(c) : c;
	}
	vec forwardPropagation_express(vec x) override {
		i = g(wi * x + ri * c + bi);
		f = g(wf * x + rf * c + bf);

		c &= f;
		c += i;

		return next ? next->forwardPropagation(c) : c;
	}
	vec backwardPropagation(vec dy) override {

		dc = dy;
		di = dc & g.d(i);
		df = dc & yt() & g.d(f);

		wi_gradientStorage -= di * x.T
		bi_gradientStorage -= di;
		ri_gradientStorage -= di * c.t();

		wf_gradientStorage -= df * x.t();
		bf_gradientStorage -= df;
		rf_gradientStorage -= df * c.t();

		dc &= f;

		return prev ? prev->backwardPropagation( /*dx*/(wi.t() * di) + (wf.t() * df) /*dx*/) : dc;

	}
	vec backwardPropagation_ThroughTime(vec dy) override {
		vec x = bpX.poll_last();
		vec y = bpY.poll_last();
		vec f = bpF.poll_last();
		vec i = bpI.poll_last();

		dc += dy + (ri.t() * di) + (rf.t() * df);
		df = dc & yt() & g.d(f);
		di = dc & g.d(i);

		wi_gradientStorage -= di * x.t();
		bi_gradientStorage -= di;
		ri_gradientStorage -= di * y.t();

		wf_gradientStorage -= df * x.t();
		bf_gradientStorage -= df;
		rf_gradientStorage -= df * y.t();

		dc &= f;

		if (prev != nullptr) {
			vec dx = (wi.t() * di) + (wf.t() * df);
			return prev->backwardPropagation_ThroughTime(dx);
		} else {
			return dy;
		}
	}

	void clearBackPropagationStorage() {
		bpX.clear();
	}
	void clearGradientStorage() {
		wi_gradientStorage.zero();
		ri_gradientStorage.zero();
		bi_gradientStorage.zero();

		wf_gradientStorage.zero();
		rf_gradientStorage.zero();
		bf_gradientStorage.zero();
	}
	void updateGradients() {
		wi += wi_gradientStorage & lr;
		ri += ri_gradientStorage & lr;
		bi += bi_gradientStorage & lr;

		wf += wf_gradientStorage & lr;
		rf += rf_gradientStorage & lr;
		bf += bf_gradientStorage & lr;
	}
};
#endif /* RECURRENTLAYER_H_ */
