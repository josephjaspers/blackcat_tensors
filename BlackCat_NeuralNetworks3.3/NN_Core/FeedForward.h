/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer.h"

namespace BC {

template<class derived>
struct FeedForward : public Layer<derived> {


public:

	 scal lr = scal(0.03); //fp_type == floating point


	mat w_gradientStorage;
	vec b_gradientStorage;

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
			w_gradientStorage(this->OUTPUTS, inputs),
			b_gradientStorage(this->OUTPUTS),
			w(this->OUTPUTS, inputs),
			b(this->OUTPUTS),
			x(inputs),
			y(this->OUTPUTS),
			dx(inputs)
			{

		w.randomize(-4, 4);
		b.randomize(-4, 4);
		w_gradientStorage.zero();
		b_gradientStorage.zero();
	}

	template<class T>
	auto forwardPropagation(const _vec<T> in) {
		auto x_t = x == in;
		return this->next().forwardPropagation(g(w * x_t + b));
	}
	template<class T>
	auto forwardPropagation_Express(_vec<T> x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	template<class T>
	auto backPropagation(const _vec<T> dy) {

		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;
		return this->prev().backPropagation(dx = w.t() * dy % gd(x));
	}
	template<class U, class V>
		auto train(const _vec<U>& x, const _vec<V>& y) {
		auto dy = this->next().train(g(w * x + b), y);

		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return (w.t() * dy % gd(x));
	}

	void updateWeights() {
		w += w_gradientStorage * lr;
		b += b_gradientStorage * lr;
		this->next().updateWeights();
	}

	void clearBPStorage() {
		w_gradientStorage.zero();
		b_gradientStorage.zero();
		this->next().clearBPStorage();
	}

	void write(std::ofstream& is) {
		is << this->INPUTS << ' ';
		is << this->OUTPUTS << ' ';
		w.write(is);
		b.write(is);
		x.write(is);
		y.write(is);
		dx.write(is);
		w_gradientStorage.write(is);
		b_gradientStorage.write(is);

	}
	void read(std::ifstream& os) {
		os >> this->INPUTS;
		os >> this->OUTPUTS;

		w.read(os);
		b.read(os);
		x.read(os);
		y.read(os);
		dx.read(os);
		w_gradientStorage.read(os);
		b_gradientStorage.read(os);

	}
	void setLearningRate(fp_type learning_rate) {
		lr = learning_rate;
		this->next().setLearningRate(learning_rate);
	}
};
}



#endif /* FEEDFORWARD_CU_ */
