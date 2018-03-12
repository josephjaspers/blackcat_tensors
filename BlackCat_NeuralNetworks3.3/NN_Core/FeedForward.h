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
	int INPUTS;
	int OUTPUTS;

	 const scal lr = scal(0.03); //fp_type == floating point


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
	FeedForward(int inputs, int outputs) :
			INPUTS(inputs), OUTPUTS(outputs),
			w_gradientStorage(outputs, inputs),
			b_gradientStorage(outputs),
			w(outputs, inputs),
			b(outputs),
			x(inputs),
			y(outputs),
			dx(inputs) {

		w.randomize(-4, 4);
		b.randomize(-4, 4);
		w_gradientStorage.zero();
		b_gradientStorage.zero();
	}

	template<class T> auto forwardPropagation(const vec_expr<T>& in) {
		auto x_ = x == in;
		return this->next().forwardPropagation(g(w * x_ + b));
	}
	template<class T> auto forwardPropagation_Express(const vec_expr<T>& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	template<class T> auto backPropagation(const vec_expr<T>& dy) {
		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return this->prev().backPropagation(dx == w.t() * dy % gd(x));
	}
	template<class U, class V>
		auto train(const vec_expr<U>& x, const vec_expr<V>& y) {

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
		is << INPUTS << ' ';
		is << OUTPUTS << ' ';
		w.write(is);
		b.write(is);
		x.write(is);
		y.write(is);
		dx.write(is);
		w_gradientStorage.write(is);
		b_gradientStorage.write(is);

	}
	void read(std::ifstream& os) {
		os >> INPUTS;
		os >> OUTPUTS;

		w.read(os);
		b.read(os);
		x.read(os);
		y.read(os);
		dx.read(os);
		w_gradientStorage.read(os);
		b_gradientStorage.read(os);

	}

	//multithreading stuff -----------------------------
	void updateWeights(const FeedForward& ff) {
		w += ff.w_gradientStorage * lr;
		b += ff.b_gradientStorage * lr;
	}
	void fastCopy(const FeedForward& ff) {
		w = ff.w;
		b = ff.b;
	}

};
}



#endif /* FEEDFORWARD_CU_ */
