/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace NN {

struct FeedForward {

	scal_view lr;

	mat_view dy;							//error
	mat_view y;							//outputs

	mat_view w;							//weights
	vec_view b;							//biases

	mat_view w_gradientStorage;		//weight gradient storage
	vec_view b_gradientStorage;		//bias gradient storage

	template<class t> auto forward_propagation(const expr::mat<t>& x) {
		return y = g(w * x + b);
	}
	template<class t> auto back_propagation(const expr::mat<t>& dy_) {
		dy = dy_;

		w_gradientStorage -= dy * x().t();
		b_gradientStorage -= dy;

		return w.t() * dy % gd(x());
	}
	template<class t> auto forward_propagation_express(const expr::mat<t>& x) const {
		return g(w * x + b);
	}
};
}
}

#endif /* FEEDFORWARD_CU_ */
