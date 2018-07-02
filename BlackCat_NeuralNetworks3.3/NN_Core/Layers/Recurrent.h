///*
// * FeedForward.cu
// *
// *  Created on: Jan 28, 2018
// *      Author: joseph
// */
//
//#ifndef Recurrent_Unit
//#define Recurrent_Unit
//
//#include "Layer.h"
//#include <mutex>
//
//namespace BC {
//namespace NN {
//template<class derived>
//struct Recurrent : public Layer<derived> {
//
///*
// *  TESTED AND APPROVED
// */
//
//public:
//
//	using Layer<derived>::sum_gradients;
//	using Layer<derived>::zero;
//	using Layer<derived>::xs;
//	using Layer<derived>::lr;
//
//	omp_unique<mat> w_gradientStorage; 		//gradient storage weights
//	omp_unique<mat> r_gradientStorage;		//gradient storage recurrent weights
//	omp_unique<vec> b_gradientStorage;		//gradienst storage bias
//	omp_unique<vec> c;
//	omp_unique<vec> dc;
//
//	bp_list<vec> ys;							//storage for outputs
//	auto& xs() { return this->prev().ys(); }	//get the storage for inputs
//
//	mat w;
//	mat r;
//	vec b;
//
//	Recurrent(int inputs) :
//			Layer<derived>(inputs),
//			w(this->OUTPUTS, this->INPUTS),
//			r(this->OUTPUTS, this->OUTPUTS),
//			b(this->OUTPUTS)
//		{
//		r.randomize(-1, 1);
//		w.randomize(-1, 1);
//		b.randomize(-1, 1);
//		init_storages();
//
//	}
//
//
//	auto forward_propagation(const vec& x) {
//		if (ys().isEmpty())
//			ys().push(c());
//
//		c() = g(w * x + r * c() + b);
//
//		ys().push(c());
//		return this->next().forward_propagation(c());
//	}
//	auto back_propagation(const vec& dy) {
//		vec& x = xs().first();				//load the last input
//		vec& c = ys().second(); ys().pop();				//load last and remove
//
//		dc() += dy;							//add the error
//		w_gradientStorage() -= dy   * x.t();
//		r_gradientStorage() -= dc() * c.t();
//		b_gradientStorage() -= dy;
//
//		return this->prev().back_propagation(w.t() * dy % gd(x));
//	}
//	auto forward_propagation_express(const vec& x) const {
//		c() = g(w * x + r * c() + b);
//		return this->next().forward_propagation_express(c());
//	}
//
//	void update_weights() {
//		//sum all the gradients
//		w_gradientStorage.for_each(sum_gradients(w, lr));
//		r_gradientStorage.for_each(sum_gradients(r, lr));
//		b_gradientStorage.for_each(sum_gradients(b, lr));
//
//		this->next().update_weights();
//	}
//
//	void clear_stored_delta_gradients() {
//		w_gradientStorage.for_each(zero);	//gradient list
//		r_gradientStorage.for_each(zero);	//gradient list
//		b_gradientStorage.for_each(zero);	//gradient list
//
//		dc.for_each([](auto& var) { var.zero(); }); 	//gradient list
//		ys.for_each([](auto& var) { var.clear();});		//bp_list
//
//		this->next().clear_stored_delta_gradients();
//	}
//	void set_omp_threads(int i) {
//		ys.resize(i);
//		dc.resize(i);
//		c.resize(i);
//		w_gradientStorage.resize(i);
//		b_gradientStorage.resize(i);
//		r_gradientStorage.resize(i);
//
//		init_storages();
//		this->next().set_omp_threads(i);
//	}
//
//	void write(std::ofstream& os) {
//		w.write(os);
//		r.write(os);
//		b.write(os);
//
//		this->next().write(os);
//	}
//	void read(std::ifstream& is) {
//		w.read(is);
//		r.read(is);
//		b.read(is);
//
//		this->next().write(is);
//	}
//	void init_storages() {
//		//for each matrix/vector gradient storage initialize to correct dims
//		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS);  var.zero(); });
//		r_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->OUTPUTS); var.zero(); });
//		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			     var.zero(); });
//
//		//for each cell-state error initialize to 0
//		dc.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
//		c.for_each([&](auto& var) { var = vec(this->OUTPUTS); var.zero(); });
//
//	}
//};
//
//}
//}
//
//
//#endif /* FEEDFORWARD_CU_ */
