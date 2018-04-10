/*
 * BC_NN_Functions.h
 *
 *  Created on: Oct 7, 2017
 *      Author: joseph
 */

#ifndef BC_NN_FUNCTIONS_H_
#define BC_NN_FUNCTIONS_H_

#include "BC_NeuralNetwork_Definitions.h"

namespace nonLin {


	void constrain(tensor& t, double l_bound, double u_bound) {
		for (int i = 0; i < t.size(); ++i) {
//			if (t.data()[i] < l_bound)
//			t.data()[i] = l_bound;
//			else if (t.data()[i] > u_bound) {
//				t.data()[i] =  u_bound;
//			}
		}
	}

//	tensor abs(tensor value) {
//		tensor absolute = value;
//
//		for (unsigned i = 0; i < absolute.size(); ++i) {
////			if (absolute.data()[i] < 0) {
////				absolute.data()[i] *= -1;
////			}
//		}
//		return absolute;
//	}

	void sigmoid(tensor& x) {
		for (unsigned i = 0; i < x.size(); ++i) {
			x.accessor().getData()[i] = 1 / (1 + pow(2.71828, -x.accessor().getData()[i]));
		}
	}

	void sigmoid_deriv(tensor& x) {
		for (unsigned i = 0; i < x.size(); ++i) {
			x.accessor().getData()[i] *= (1 - x.accessor().getData()[i]);
		}
	}
};

#endif /* BC_NN_FUNCTIONS_H_ */
