/*
 * ConvLayer.h
 *
 *  Created on: Mar 27, 2018
 *      Author: joseph
 */

#ifndef CONVLAYER_H_
#define CONVLAYER_H_

#include "Layer.h"

namespace BC {

template<class derived>
struct Conv : public Layer<derived> {


public:
	 const scal lr = scal(0.03); //fp_type == floating point

	cube w_gradientStorage;
	cube w;

	cube img;
	cube y;
	cube x;
	/*
	 * *Note: the operator == represents a delayed evaluation assignment operator.
	 * 	It is mathematically equivalent to the operator= (copy operator) however it does not get evaluated until an
	 * 	actual operator=, this allows for chaining together multiple complex assignment expressions.
	 * 	It also allows for passing an expression with an assignment and delaying the operation for more optimizations
	 *
	 */
	Conv(int row, int cols, int z) :
			x(row, cols, z),
			w_gradientStorage(3, 3, 10),
			w(outputs, inputs),
			y(),
			dx(inputs) {

		w.randomize(-1, 1);
		w_gradientStorage.zero();
	}

	template<class T> auto forwardPropagation(const vec_expr<T>& in) {
		auto x_t = x == in;
		return this->next().forwardPropagation(g(w * x_t + b));
	}
	template<class T> auto forwardPropagation_Express(const vec_expr<T>& x) const {
		return this->next().forwardPropagation_Express(g(w * x + b));
	}

	template<class T> auto backPropagation(const vec_expr<T>& dy) {
		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;

		return this->prev().backPropagation(dx = w.t() * dy % gd(x));
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
}
};


#endif /* CONVLAYER_H_ */
