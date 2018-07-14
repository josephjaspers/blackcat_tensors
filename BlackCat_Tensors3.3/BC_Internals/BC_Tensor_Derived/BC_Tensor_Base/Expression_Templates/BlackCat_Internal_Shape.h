/*
 * Shape.h
 *
 *  Created on: May 24, 2018
 *      Author: joseph
 */

#ifndef INTERNAL_SHAPE_H_
#define INTERNAL_SHAPE_H_

#include <type_traits>
#include "BlackCat_Internal_Definitions.h"

namespace BC {

template<int dimension>
struct Shape {

	__BCinline__ static constexpr int LENGTH() { return dimension; }
private:
	static constexpr int last = dimension - 1;
	int IS[dimension];
	int OS[dimension];
public:

	__BCinline__ int size() const { return LENGTH() > 0 ?  OS[last] : 1; }

	__BCinline__ const auto inner_shape() const { return ptr_array<dimension>(IS); }
	__BCinline__ const auto outer_shape() const { return ptr_array<dimension>(OS); }


	Shape() = default;
	Shape(const Shape&) = default;
	Shape(Shape&&) = default;
	Shape& operator = (const Shape&) = default;
	Shape& operator = (Shape&&) = default;

	template<class... integers> Shape(int first, integers... ints) : IS() {
		static_assert(sizeof...(integers) + 1 == dimension, "integer initialization must have the same number of dimensions");
		this->init(first, ints...);
	}
	template<int dim, class int_t>
	Shape (stack_array<dim, int_t> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		if (LENGTH() > 0) {
			IS[0] = param[0];
			OS[0] = IS[0];
			for (int i = 1; i < LENGTH(); ++i) {
				IS[i] = param[i];
				OS[i] = OS[i - 1] * IS[i];
			}
		}
	}
	template<int dim, class int_t>
	Shape (pointer_array<dim, int_t> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		if (LENGTH() > 0) {
			IS[0] = param[0];
			OS[0] = IS[0];
			for (int i = 1; i < LENGTH(); ++i) {
				IS[i] = param[i];
				OS[i] = OS[i - 1] * IS[i];
			}
		}
	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		if (LENGTH() > 0) {
			IS[0] = param[0];
			OS[0] = IS[0];
			for (int i = 1; i < LENGTH(); ++i) {
				IS[i] = param[i];
				OS[i] = OS[i - 1] * IS[i];
			}
		}
	}

private:
	template<int d = 0> __BCinline__
	void init() {}

	template<int dim = 0, class... integers> __BCinline__
	void init(int front, integers... ints) {

		//NVCC gives warning if you convert the static_assert into a one-liner
		static constexpr bool intList = MTF::is_integer_sequence<integers...>;
		static_assert(intList, "MUST BE INTEGER LIST");

		IS[dim] = front;

		if (dim > 0)
			OS[dim] = front * outer_shape()[dim - 1];
		else
			OS[0] = front;

		if (dim != LENGTH() - 1) {
			init<(dim + 1 < LENGTH() ? dim + 1 : LENGTH())>(ints...);
		}
	}
};

template<>
struct Shape<0> {
	__BCinline__ int size() const { return 1; }
	__BCinline__ const auto inner_shape() const { return l_array<0>([&](auto x) { return 1; });}
	__BCinline__ const auto outer_shape() const { return l_array<0>([&](auto x) { return 0; });}
};

template<>
struct Shape<1> {
	static constexpr int dimension = 1;
	int length;

	Shape(int length_) : length(length_) {}

	template<int dim, class int_t>
	Shape (stack_array<dim, int_t> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];
	}
	template<int dim, class int_t>
	Shape (pointer_array<dim, int_t> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];

	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];

	}

	__BCinline__ int size() const { return length; }
	__BCinline__ const auto inner_shape() const { return l_array<1>([&](auto x) { return length; });}
	__BCinline__ const auto outer_shape() const { return l_array<1>([&](auto x) { return x == 0 ? length : 0; });}
};
}



#endif /* SHAPE_H_ */
