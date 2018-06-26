/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef GRU_UNIT
#define GRU_UNIT

#include "Layer.h"
#include <mutex>

namespace BC {
namespace NN {
template<class derived>
struct GRU : public Layer<derived> {

/*
 *  TESTED AND APPROVED
 */

public:

	using Layer<derived>::xs;
	using Layer<derived>::lr;

	omp_unique<mat> wz_gradientStorage, wf_gradientStorage; 	//gradient storage weights
	omp_unique<mat> rz_gradientStorage, rf_gradientStorage;		//gradient storage recurrent weights
	omp_unique<vec> bz_gradientStorage, bf_gradientStorage;		//gradienst storage bias
	omp_unique<vec> c,  f,  z;
	omp_unique<vec> dc, df, dz;
	bp_list<vec> 	ys, fs, zs;							//storage for outputs
	auto& xs() { return this->prev().ys(); }	//get the storage for inputs

	mat wz, wf;
	mat rz, rf;
	vec bz, bf;

	GRU(int inputs) :
			Layer<derived>(inputs),
			wz(this->OUTPUTS, this->INPUTS),
			rz(this->OUTPUTS, this->OUTPUTS),
			bz(this->OUTPUTS),
			wf(this->OUTPUTS, this->INPUTS),
			rf(this->OUTPUTS, this->OUTPUTS),
			bf(this->OUTPUTS)
		{
		rz.randomize(-1, 1);
		wz.randomize(-1, 1);
		bz.randomize(-1, 1);
		rf.randomize(-1, 0);
		wf.randomize(-1, 0);
		bf.randomize(-1, 0);
		init_storages();

	}


	auto forwardPropagation(const vec& x) {
		if (zs().isEmpty()) {
			zs().push(z());
			fs().push(f());
			ys().push(c());
		}



		f() = g(wf * x + rf * f() + bf);
		c() = c() % f() + z();

		zs().push(z());
		fs().push(f());
		ys().push(c());
		return this->next().forwardPropagation(c());
	}
	auto backPropagation(const vec& dy) {
		vec& x = xs().first();
		vec& c = ys().second(); ys().pop();
		vec& f = fs().second();
		vec  F = fs().pop();
		vec& z = zs().second(); zs().pop();

		dc() = dc() % F + dy + rz.t() * dz() + rf.t() * df();
		dz() = dc() % gd(z);
		df() = dc() % c % gd(f);

		wz_gradientStorage() -= dz() * x.t();
		rz_gradientStorage() -= dz() * z.t();
		bz_gradientStorage() -= dz();

		wf_gradientStorage() -= df() * x.t();
		rf_gradientStorage() -= df() * f.t();
		bf_gradientStorage() -= df();

		return this->prev().backPropagation((wz.t() * dz()  + wf.t() * df()) % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		z() = g(wz * x + rz * z() + bz);
		f() = g(wf * x + rf * f() + bf);
		c() = c() % f() + z();
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

		dc.for_each([](auto& var) { var.zero(); }); 	//gradient list
		df.for_each([](auto& var) { var.zero(); }); 	//gradient list
		dz.for_each([](auto& var) { var.zero(); }); 	//gradient list

		ys.for_each([](auto& var) { var.clear();});		//bp_list
		fs.for_each([](auto& var) { var.clear();});		//bp_list
		zs.for_each([](auto& var) { var.clear();});		//bp_list

		this->next().clearBPStorage();
	}
	void set_omp_threads(int i) {
		dc.resize(i);
		df.resize(i);
		dz.resize(i);

		c.resize(i);
		f.resize(i);
		z.resize(i);

		wz_gradientStorage.resize(i);
		bz_gradientStorage.resize(i);
		rz_gradientStorage.resize(i);
		wf_gradientStorage.resize(i);
		bf_gradientStorage.resize(i);
		rf_gradientStorage.resize(i);

		ys.resize(i);
		fs.resize(i);
		zs.resize(i);

		init_storages();
		this->next().set_omp_threads(i);
	}

	void write(std::ofstream& os) {
//		os << this->INPUTS << ' ';
//		os << this->OUTPUTS << ' ';
		wz.write(os);
		wf.write(os);
		rz.write(os);
		rf.write(os);
		bz.write(os);
		bf.write(os);

		this->next().write(os);
	}
	void read(std::ifstream& is) {
//		is >> this->INPUTS;
//		is >> this->OUTPUTS;
		wz.read(is);
		wf.read(is);
		rz.read(is);
		rf.read(is);
		bz.read(is);
		bf.read(is);

		this->next().read(is);
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
};

}
}


#endif /* FEEDFORWARD_CU_ */
