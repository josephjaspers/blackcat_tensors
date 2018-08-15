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
#include "Shape_Base.h"
#include <array>
namespace BC {
template<class T, int dims>
struct simple_array {
	int value[dims];
	__BCinline__ int& operator[] (int i ) { return value[i]; }
	__BCinline__ const int& operator[] (int i ) const { return value[i]; }

};

template<int dims>
struct Shape : Shape_Base<dims, Shape<dims>> {

	__BCinline__ static constexpr int LENGTH() { return dims; }
protected:
	static constexpr int last = dims - 1;

	simple_array<int, dims> IS;
	simple_array<int, dims> OS;
public:

	void swap_shape(Shape& b) {
		std::swap(IS, b.IS);
		std::swap(OS, b.OS);
	}

	Shape(const Shape&) = default;
	Shape(Shape&&) = default;

	template<class... integers> Shape(integers... ints) : IS() {
		static_assert(MTF::is_integer_sequence<integers...>, "INTEGER LIST OF SHAPE");
		static_assert(sizeof...(integers) == dims, "integer initialization must have the same number of dimensions");
		init(BC::array(ints...));
	}
	Shape() {};

	template<class is_deriv>
	Shape(const Inner_Shape<dims, is_deriv> param) {
		init(param);
	}

	template<int dim, class int_t>
	Shape (stack_array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	template<int dim, class int_t>
	Shape (simple_array<int_t, dim> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	__BCinline__ const auto inner_shape() const { return IS; }
	__BCinline__ const auto outer_shape() const { return OS; }
	__BCinline__ const auto block_shape() const { return ptr_array<dims>(OS); }

	__BCinline__ int size() const { return OS[last]; }
	__BCinline__ int rows() const { return IS[0]; }
	__BCinline__ int cols() const { return IS[1]; }
	__BCinline__ int dimension(int i) const { return IS[i]; }
	__BCinline__ int outer_dimension() const { return IS[dims - 2]; }
	__BCinline__ int leading_dimension(int i) const { return OS[i]; }

//	void copy_shape(const Shape& shape) {
//		for (int i = 0; i < dims; ++i) {
//			IS[i] = shape.IS[i];
//			OS[i] = shape.OS[i];
//		}
//	}
	template<class T>
	void copy_shape(const Shape_Base<dims, T>& shape) {
		for (int i = 0; i < dims; ++i) {
			IS[i] = shape.dimension(i);
			OS[i] = shape.leading_dimension(i);
		}
	}


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

	template<class deriv> void copy_shape(const Shape_Base<0, deriv>& shape) {}
	static void swap_shape(Shape& a, Shape& b) {}

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

	void copy_shape(const Shape<1>& shape) {
		this->length = shape.length;
	}

	template<class deriv> void copy_shape(const Shape_Base<1, deriv>& shape) {
		this->length = shape.dimension(0);
	}

	void swap_shape(Shape<1>& shape){
		std::swap(length, shape.length);
	}

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
	Shape (simple_array<int_t, dim> param) {
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
