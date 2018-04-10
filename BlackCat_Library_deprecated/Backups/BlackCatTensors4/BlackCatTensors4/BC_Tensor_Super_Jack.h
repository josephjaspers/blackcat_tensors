/*
 * BC_Tensor_Super_Queen.h
 *
 *  Created on: Nov 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_JACK_H_
#define BC_TENSOR_SUPER_JACK_H_

#include "BC_Tensor_Super_Queen.h"

//Will move this to another class later
namespace printHelper {
	template<int curr, int ... stack>
	struct f {
		void fill(int* ary) {
			ary[0] = curr;
			f<stack...>().fill(&ary[1]);
		}
	};
	template<int dim>
	struct f<dim> {
		void fill(int* ary) {
			ary[0] = dim;
		}
	};

}

/*
 * Tensor Queen defines utility methods for primary Tensors
 */

template<typename T, class ml = CPU, int ... dimensions>
class Tensor_Jack : public Tensor_Queen<T, ml, dimensions...> {
public:
	/*
	 *Tensor_Queen defines core methods
	 */

	void zero() {
		ml::zero(this->array, this->size());
	}

	void fill(T value) {
		ml::fill(this->array, value, this->size());
	}
	void randomize(T lb, T ub) {
		ml::randomize(this->array, lb, ub, this->size());
	}

	void print() const {
		int* ranks = new int[sizeof...(dimensions)];
		printHelper::f<dimensions...>().fill(ranks);
		ml::print(this->array, ranks, sizeof...(dimensions), 5);
	}

};
#endif /* BC_TENSOR_SUPER_JACK_H_ */
