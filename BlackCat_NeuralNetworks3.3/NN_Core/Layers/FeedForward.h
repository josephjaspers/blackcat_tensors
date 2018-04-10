/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer.h"
#include <mutex>

namespace BC {

template<class derived>
struct FeedForward : public Layer<derived> {


public:

	 scal lr = scal(0.03); //fp_type == floating point

	 gradient_list<mat> w_gradientStorage = gradient_list<mat>(1);
	 gradient_list<vec> b_gradientStorage = gradient_list<vec>(1);

	bp_list<vec> xs = bp_list<vec>(8);

	mat w;

	vec x;
	vec y;
	vec b;

	vec dx;

	/*
	 * *Note: the operator == represents a delayed evaluation assignment operator.
	 * 	It is mathematically equivalent to the operator= (copy operator) however it does not get evaluated until an
	 * 	actual operator=, this allows for chaining together multiple complex assignment expressions.
	 * 	It also allows for passing an expression with an assignment and delaying the operation for more optimizations
	 *
	 */
	FeedForward(int inputs) :
			Layer<derived>(inputs),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS),
			x(inputs),
			y(this->OUTPUTS),
			dx(inputs)
			{

		w.randomize(-4, 4);
		b.randomize(-4, 4);
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			   });
	}

	vec forwardPropagation(const vec& x_) {
		xs().push_front(std::move(vec(x_)));	//store the inputs
		auto& x = xs().front();					//load what we just stored

		return this->next().forwardPropagation(g(w * x + b));
	}
	vec backPropagation(const vec& dy_) {
		vec dy = dy_;
		vec x = xs().pop_front();				//load the last input

		w_gradientStorage() -= dy * x.t();		//does work
		b_gradientStorage() -= dy;
		return this->prev().backPropagation(dx = w.t() * dy % gd(x));
	}
	auto forwardPropagation_Express(const vec& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	void updateWeights() {
		xs.clear();
		w_gradientStorage.for_each([&](auto& var) { w += var * lr; });
		b_gradientStorage.for_each([&](auto& var) { b += var * lr; });
		this->next().updateWeights();
	}

	void clearBPStorage() {
		xs.clear();
		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); var.zero(); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			    var.zero(); });
		this->next().clearBPStorage();
	}
	void init_threads(int i) {
		xs.resize(i);
		w_gradientStorage.resize(i);
		b_gradientStorage.resize(i);

		w_gradientStorage.for_each([&](auto& var) { var = mat(this->OUTPUTS, this->INPUTS); var.zero(); });
		b_gradientStorage.for_each([&](auto& var) { var = vec(this->OUTPUTS);			    var.zero(); });
		this->next().init_threads(i);
	}

	void write(std::ofstream& is) {
		is << this->INPUTS << ' ';
		is << this->OUTPUTS << ' ';
		w.write(is);
		b.write(is);
		x.write(is);
		y.write(is);
		dx.write(is);
	}
	void read(std::ifstream& os) {
		os >> this->INPUTS;
		os >> this->OUTPUTS;

		w.read(os);
		b.read(os);
		x.read(os);
		y.read(os);
		dx.read(os);
	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};
}
//	template<class T>
//	auto forwardPropagation(const _vec<T> in) {
//		auto x_t = x == in;
//		return this->next().forwardPropagation(g(w * x_t + b));
//	}
//	template<class T>
//	auto backPropagation(const _vec<T> dy) {
//
//		w_gradientStorage -= dy * x.t();
//		b_gradientStorage -= dy;
//		return this->prev().backPropagation(dx = w.t() * dy % gd(x));
//	}

#endif /* FEEDFORWARD_CU_ */
