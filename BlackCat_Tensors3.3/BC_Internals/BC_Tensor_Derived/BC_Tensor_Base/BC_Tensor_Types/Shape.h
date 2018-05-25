/*
 * Shape.h
 *
 *  Created on: May 24, 2018
 *      Author: joseph
 */

#ifndef INTERNAL_SHAPE_H_
#define INTERNAL_SHAPE_H_

namespace BC {

template<int dimension>
struct Shape {

	static constexpr int last = dimension - 1;
	__BCinline__ static constexpr int LENGTH() { return dimension; }

	int inner_shape[dimension];
	int outer_shape[dimension];

	__BCinline__ int size() const { return LENGTH() > 0 ?  outer_shape[last] : 1; }

	__BCinline__ int* is() { return inner_shape; }
	__BCinline__ const int* is() const { return inner_shape; }

	__BCinline__ int* os() { return outer_shape; }
	__BCinline__ const int* os() const { return outer_shape; }


	Shape() = default;

	template<class... integers> Shape(int first, integers... ints) : inner_shape() {
		this->init(first, ints...);
	}

	template<class U>
	Shape (const U& param) {
		if (LENGTH() > 0) {
			inner_shape[0] = param[0];
			outer_shape[0] = inner_shape[0];
			for (int i = 1; i < LENGTH(); ++i) {
				inner_shape[i] = param[i];
				outer_shape[i] = outer_shape[i - 1] * inner_shape[i];
			}
		}
	}

	template<int d = 0> __BCinline__
	void init() {}

	template<int dim = 0, class... integers> __BCinline__
	void init(int front, integers... ints) {

		//NVCC gives warning if you convert the static_assert into a one-liner
		static constexpr bool intList = MTF::is_integer_sequence<integers...>;
		static_assert(intList, "MUST BE INTEGER LIST");

		is()[dim] = front;

		if (dim > 0)
			os()[dim] = front * os()[dim - 1];
		else
			os()[0] = front;

		if (dim != LENGTH() - 1) {
			init<(dim + 1 < LENGTH() ? dim + 1 : LENGTH())>(ints...);
		}
	}

};
}



#endif /* SHAPE_H_ */
