/*
 * RecurrentLayer.h
 *
 *  Created on: Aug 21, 2017
 *      Author: joseph
 */

#ifndef BLACKCAT_LSTM
#define BLACKCAT_LSTM

class LSTM_Unit : public Layer {
	nonLinearityFunction g;
	nonLinearityFunction h;

	vec c;				//cellstate
	vec dc;				//error of cell state
	vec x, dx;

	vec y;				//current output (or output during BP)

	vec f, i, z, o;		//current activations for forget, input, write, output gate

	vec bz, bi, bf, bo;	//Bias Vectors for gate
	vec dz, di, df, od;	//error of each gate
	mat wz, wi, wf, wo;	//feed forward weights of each gate
	mat rz, ri, rf, ro;	//recurrent weights of each gate

	vec bz_gradientStorage, bi_gradientStorage, bf_gradientStorage, bo_gradientStorage; //Bias gradient storage of gates
	mat wz_gradientStorage, wi_gradientStorage, wf_gradientStorage, wo_gradientStorage; //Weight gradient storage of gates
	mat rz_gradientStorage, ri_gradientStorage, rf_gradientStorage, ro_gradientStorage; //recurrnet gradient storage of gates

	bpStorage bpZ, bpI, bpF, bpO, bpC;

public:
	LSTM_Unit(unsigned inputs, unsigned outputs) {

		z = i = f = o = vec(outputs);

		dc = vec(outputs);
		dx = vec(inputs);

		wz = wi = wf = wo = mat(outputs, inputs);
		rz = ri = rf = ro = mat(outputs, outputs);
		bz = bi = bf = bo = vec(outputs);

		wz_gradientStorage = wi_gradientStorage = wf_gradientStorage = wo_gradientStorage = mat(outputs, inputs);
		rz_gradientStorage = ri_gradientStorage = rf_gradientStorage = ro_gradientStorage = mat(outputs, inputs);
		bz_gradientStorage = bi_gradientStorage = bf_gradientStorage = bo_gradientStorage = vec(outputs);

		ri.randomize(-3, 3);
		bi.randomize(-3, 3);
		wi.randomize(-3, 3);

		rf.randomize(-3, 0);
		bf.randomize(-3, 0);
		wf.randomize(-3, 0);
	}
	virtual ~LSTM_Unit() {

	}
	vec forwardPropagation(vec x) override {
		bpF.store(f);
		bpZ.store(z);
		bpI.store(i);
		bpO.store(o);
		bpC.store(c);

		f = g(wf * x + rf * y + bf);
		z = g(wz * x + rz * y + bz);
		i = g(wi * x + ri * y + bi);
		o = g(wo * x + ro * y + bo);

		c &= f;
		c += (z & i);

		y = g(c) & o; //g -parenthesis operator applies the nonlinearity function to a reference to the parameter, nonLin creates and returns a copy. (This preserves the original cell state)

		//continue forwardprop
		if (next != nullptr)
			return next->forwardPropagation(y);
		else
			return y;
	}
	vec forwardPropagation_express(vec x) override {
		f = g(wf * x + rf * y + bf);
		z = g(wz * x + rz * y + bz);
		i = g(wi * x + ri * y + bi);
		o = g(wo * x + ro * y + bo);

		c &= f;
		c += (z & i);

		y = (g(c) & o);

		return next != nullptr ? next->forwardPropagation(c) : c;
	}
	vec backwardPropagation(vec dy) override {
		tensor& ct = bpY.last();

		dc = dy & o & g.d(g(c));
		od = dy & g(c) & g.d(o);
		df = dc & ct & g.d(f);
		dz = dc & i & g.d(z);
		di = dc & z & g.d(i);
		dc &= f;
		//Store gradients
		//calculate input error
		storeGradients();

		vec dx = (wz.t() * dz + wf.t() * df + wi.t() * di + wo.t() * od);

		if (prev != nullptr) {
			return prev->backwardPropagation(dx);
		} else {
			return dy;
		}
	}
	vec backwardPropagation_ThroughTime(vec dy) override {
		vec c = bpC.poll_last();
		vec z = bpZ.poll_last();
		vec f = bpF.poll_last();
		vec i = bpI.poll_last();
		vec o = bpO.poll_last();
		tensor& ct = bpC.last();

		dc += dy + rz.t() * dz + ri.t() * di + rf.t() * df + ro.t() * od;
		//math of error
		dc += dy & g.d(y) & o & g.d(g(c));
		od = dc & g(c) & g.d(o);
		df = dc & ct & g.d(f);
		dz = dc & i & g.d(z);
		di = dc & z & g.d(i);
		//Store gradients
		wz_gradientStorage -= dz * x;
		bz_gradientStorage -= dz;
		rz_gradientStorage -= dz * c;

		wf_gradientStorage -= df * x;
		bf_gradientStorage -= df;
		rf_gradientStorage -= df * c;

		wi_gradientStorage -= di * x;
		bi_gradientStorage -= di;
		ri_gradientStorage -= di * c;

		wo_gradientStorage -= od * x;
		bo_gradientStorage -= od;
		ro_gradientStorage -= od * c;
		//get input error
		vec dx = (wz.t() * dz) + (wf.t() * df) + (wi.t() * di) + (wo.t() * od);
		//send the error through the gate
		dc &= bpF.last();
		//update backprop storage

		if (prev != nullptr) {
			return prev->backwardPropagation(dx);
		} else {
			return dy;
		}
	}

	void clearBackPropagationStorage() {
		bpZ.clear();
		bpI.clear();
		bpF.clear();
		bpO.clear();
		bpY.clear();
		bpX.clear();
	}
	void clearGradientStorage() {
		dc.zero();

		wi_gradientStorage.zero();
		ri_gradientStorage.zero();
		bi_gradientStorage.zero();

	}
	void updateGradients() {
		wz += wz_gradientStorage & lr;
		rz += rz_gradientStorage & lr;
		bz += bz_gradientStorage & lr;

		wi += wi_gradientStorage & lr;
		ri += ri_gradientStorage & lr;
		bi += bi_gradientStorage & lr;

		wf += wf_gradientStorage & lr;
		rf += rf_gradientStorage & lr;
		bf += bf_gradientStorage & lr;

		wo += wo_gradientStorage & lr;
		ro += ro_gradientStorage & lr;
		bo += bo_gradientStorage & lr;
	}

	void storeGradients() {
		wz_gradientStorage -= dz * x;
		bz_gradientStorage -= dz;
		rz_gradientStorage -= dz * c;

		wf_gradientStorage -= df * x;
		bf_gradientStorage -= df;
		rf_gradientStorage -= df * c;

		wi_gradientStorage -= di * x;
		bi_gradientStorage -= di;
		ri_gradientStorage -= di * c;

		wo_gradientStorage -= od * x;
		bo_gradientStorage -= od;
		ro_gradientStorage -= od * c;
	}
};
#endif /* GatedRecurrentUnit */

