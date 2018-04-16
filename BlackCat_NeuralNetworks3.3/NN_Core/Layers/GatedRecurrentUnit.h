/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef BC_GATEDRECURRENT_Unit
#define BC_GATEDRECURRENT_Unit

#include "Layer.h"

namespace BC {
namespace NN {


template<class derived>
struct GRU : public Layer<derived> {

public:

	using Layer<derived>::sum_gradients;
	using Layer<derived>::zero;
	using Layer<derived>::clear;
	using Layer<derived>::xs;
	using Layer<derived>::lr;

	omp_unique<mat> wz_gradientStorage; 	//gradient storage weights
	omp_unique<mat> rz_gradientStorage;		//gradient storage recurrent weights
	omp_unique<vec> bz_gradientStorage;		//gradienst storage bias

	omp_unique<mat> wf_gradientStorage; 	//gradient storage forget weights
	omp_unique<mat> rf_gradientStorage;		//gradient storage forget recurrent weights
	omp_unique<vec> bf_gradientStorage;		//gradienst storage forget bias

	omp_unique<vec> c, dc;
	omp_unique<vec> f, df;
	omp_unique<vec> z, dz;

	bp_list<vec> ys, fs, zs;

	mat wz, wf;
	mat rz, rf;
	vec bz, bf;

	auto& xs() { return this->prev().ys(); }	//get the storage for inputs



	GRU(int inputs) :
			Layer<derived>(inputs),
			wf(this->OUTPUTS, this->INPUTS),
			rf(this->OUTPUTS, this->OUTPUTS),
			bf(this->OUTPUTS),
			wz(this->OUTPUTS, this->INPUTS),
			rz(this->OUTPUTS, this->OUTPUTS),
			bz(this->OUTPUTS)
			{
		rf.randomize(-4, 0);
		wf.randomize(-4, 4);
		bf.randomize(-4, 4);
		rz.randomize(-4, 0);
		wz.randomize(-4, 4);
		bz.randomize(-4, 4);

		init_storages();
	}


	vec forwardPropagation(const vec& x) {
		fp_prep();

		f() = g(wf * x + rf * f() + bf);
		z() = g(wz * x + rz * z() + bz);
		c() = c() ** f() + z();

		xs().push(x);
		fs().push(f());
		zs().push(z());
		return this->next().forwardPropagation(c());
	}

	vec backPropagation(const vec& dy) {
		vec& x = xs().first();				//load the last input
		vec& c = ys().second();				//load last and remove
		vec f = fs().pop();
		vec z = zs().pop();
				ys().pop();							//update

		dc() += dy + rz.t() * dz() + rf.t() * df();
		df() == dc() ** c ** gd(f);
		dz() == dc() ** gd(z);

		wz_gradientStorage() -= dy   * x.t();
		rz_gradientStorage() -= dz() * z.t();
		bz_gradientStorage() -= dy;

		wf_gradientStorage() -= dy   * x.t();
		rf_gradientStorage() -= df() * f.t();
		bf_gradientStorage() -= dy;

		dc() = dc() ** f;

		vec dx = (wz.t() * dy + wf.t() * dy) ** gd(x);
		return this->prev().backPropagation(dx);
	}
	auto forwardPropagation_Express(const vec& x) const {
		f() = g(wf * x + rf * c() + bf);
		z() = g(wz * x + rz * c() + bz);
		c() = c() ** f() + z();
		return this->next().forwardPropagation_Express(c());
	}

	void updateWeights() {
		//sum all the gradients
		wz_gradientStorage.for_each(sum_gradients(wz, lr));
		rz_gradientStorage.for_each(sum_gradients(rz, lr));
		bz_gradientStorage.for_each(sum_gradients(bz, lr));

		wf_gradientStorage.for_each(sum_gradients(wf, lr));
		rf_gradientStorage.for_each(sum_gradients(rf, lr));
		bf_gradientStorage.for_each(sum_gradients(bf, lr));

		this->next().updateWeights();
	}

	void clearBPStorage() {
		wz_gradientStorage.for_each(zero);	//gradient list
		rz_gradientStorage.for_each(zero);	//gradient list
		bz_gradientStorage.for_each(zero);	//gradient list

		wf_gradientStorage.for_each(zero);	//gradient list
		rf_gradientStorage.for_each(zero);	//gradient list
		bf_gradientStorage.for_each(zero);	//gradient list

		dc.for_each(zero);
		df.for_each(zero);
		dz.for_each(zero);
		c.for_each(zero);
		f.for_each(zero);
		z.for_each(zero);

		ys.for_each(clear);
		fs.for_each(clear);
		zs.for_each(clear);

		this->next().clearBPStorage();
	}
	void set_omp_threads(int i) {
		ys.resize(i);
		dc.resize(i);
		c.resize(i);
		wz_gradientStorage.resize(i);
		bz_gradientStorage.resize(i);
		rz_gradientStorage.resize(i);
		wf_gradientStorage.resize(i);
		bf_gradientStorage.resize(i);
		rf_gradientStorage.resize(i);

		init_storages();
	}

	void write(std::ofstream& is) {
	}
	void read(std::ifstream& os) {
	}
	void init_storages() {
		//for each matrix/vector gradient storage initialize to correct dims
		wz_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS);  var.zero(); });
		rz_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->OUTPUTS); var.zero(); });
		bz_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			      var.zero(); });

		wf_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS);  var.zero(); });
		rf_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->OUTPUTS); var.zero(); });
		bf_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			      var.zero(); });

		//for each cell-state error initialize to 0
		dc.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
		df.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
		dz.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });

		c.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
		f.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
		z.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
	}
	void fp_prep() {
		//Store the original or "first" time stamp for back prop training.
		//If there are already values than this has been already called
		if (zs().isEmpty()) zs().push(z());
		if (fs().isEmpty()) fs().push(z());
		if (ys().isEmpty()) ys().push(z());

	}

};
}
}

#endif
