/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer.cu"

namespace BC {

struct FeedForward : Layer<FeedForward> {

public:
	int INPUTS;
	int OUTPUTS;

	mat w_gradientStorage;
	vec b_gradientStorage;

	mat w;

	vec x;
	vec y;
	vec b;

	vec dx;

	//operator == is a delayed evaluation assignment operator
	FeedForward(int inputs, int outputs) :
			INPUTS(inputs), OUTPUTS(outputs),
			w_gradientStorage(outputs, inputs),
			b_gradientStorage(outputs),
			w(outputs, inputs),
			b(outputs),
			x(inputs),
			y(outputs),
			dx(inputs) {

		w.randomize(-3, 3);
		b.randomize(-3, 3);
		w_gradientStorage.zero();
		b_gradientStorage.zero();
	}
	FeedForward(std::ifstream& is) {
		is >> INPUTS;
		is >> OUTPUTS;

		w.read(is);
		b.read(is);
		x.read(is);
		y.read(is);
		dx.read(is);
		w_gradientStorage.read(is);
		b_gradientStorage.read(is);

	}

	template<class T, class ML> auto forwardPropagation(vec_expr<T, ML> in) {
		auto x_t = x == in;
		//creates a temporary that states evaluate x to in when appropriate
		return y = g(w * x_t + b);

	}
	template<class T, class ML> auto forwardPropagation_Express(vec_expr<T,ML> x) const {
		return g(w * x + b);
	}

	template<class T, class ML> auto backPropagation(vec_expr<T, ML> dy) {
		w_gradientStorage -= dy * x.t();
		b_gradientStorage -= dy;
		return dx = w.t() * dy % gd(x);							//** is point_wise multiply
	}

	void updateWeights() {
		w += w_gradientStorage * lr;
		b += b_gradientStorage * lr;
	}
	void clearBPStorage() {
		w_gradientStorage.zero();
		b_gradientStorage.zero();
	}

	int getClass() {
		return LAYER_TYPE::FeedForward_;
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

};
}



#endif /* FEEDFORWARD_CU_ */
