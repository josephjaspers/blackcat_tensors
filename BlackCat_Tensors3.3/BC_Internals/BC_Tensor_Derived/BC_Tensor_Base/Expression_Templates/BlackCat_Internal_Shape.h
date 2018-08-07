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

template<int dims>
struct Shape {

	__BCinline__ static constexpr int LENGTH() { return dims; }
private:
	static constexpr int last = dims - 1;
	int IS[dims];
	int OS[dims];
public:

	template<class... integers> Shape(integers... ints) : IS() {
		static_assert(sizeof...(integers) == dims, "integer initialization must have the same number of dimensions");
		init(BC::array(ints...));
	}
	Shape() {};
	template<int dim, class int_t>
	Shape (stack_array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	template<int dim, class int_t>
	Shape (pointer_array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	__BCinline__ const auto inner_shape() const { return ptr_array<dims>(IS); }
	__BCinline__ const auto outer_shape() const { return ptr_array<dims>(OS); }
	__BCinline__ const auto block_shape() const { return ptr_array<dims>(OS); }

	__BCinline__ int size() const { return OS[last]; }
	__BCinline__ int rows() const { return IS[0]; }
	__BCinline__ int cols() const { return IS[1]; }
	__BCinline__ int dimension(int i) const { return IS[i]; }
	__BCinline__ int outer_dimension() const { return IS[dims - 2]; }
	__BCinline__ int leading_dimension(int i) const { return OS[i]; }


private:

	template<class shape_t> __BCinline__
	void init(const shape_t& param) {
		if (LENGTH() > 0) {
			IS[0] = param[0];
			OS[0] = IS[0];
			for (int i = 1; i < LENGTH(); ++i) {
				IS[i] = param[i];
				OS[i] = OS[i - 1] * IS[i];
			}
		}
	}
};

template<>
struct Shape<0> {
	__BCinline__ int size() const { return 1; }
	__BCinline__ const auto inner_shape() const { return l_array<0>([&](auto x) { return 1; });}
	__BCinline__ const auto outer_shape() const { return l_array<0>([&](auto x) { return 0; });}
	__BCinline__ int rows() const { return 1; }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return 1; }
	__BCinline__ int outer_dimension() const { return 1; }
	__BCinline__ int leading_dimension(int i) const { return 0; }
};

template<>
struct Shape<1> {
	static constexpr int dims = 1;
	int length;

	Shape(int length_) : length(length_) {}
	__BCinline__ int size() const { return length; }
	__BCinline__ int rows() const { return length; }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return (&length)[i]; }
	__BCinline__ int outer_dimension() const { return length; }
	__BCinline__ int leading_dimension(int i) const { return i == 0 ? length : 0; }

	template<int dim, class int_t>
	Shape (stack_array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];
	}
	template<int dim, class int_t>
	Shape (pointer_array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];

	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		length = param[0];

	}

	__BCinline__ const auto inner_shape() const { return l_array<1>([&](auto x) { return length; });}
	__BCinline__ const auto outer_shape() const { return l_array<1>([&](auto x) { return x == 0 ? length : 0; });}
};
}



#endif /* SHAPE_H_ */
