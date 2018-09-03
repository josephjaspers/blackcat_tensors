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

template<int dims>
struct Shape : Shape_Base<Shape<dims>> {
protected:

	BC::array<dims, int> m_inner_shape;
	BC::array<dims, int> m_outer_shape;

public:

	Shape()	= default;
	Shape(const Shape&) = default;
	Shape(		Shape&&) = default;

	template<class... integers> Shape(integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "INTEGER LIST OF SHAPE");
		static_assert(sizeof...(integers) == dims, "integer initialization must have the same number of dimensions");
		init(BC::make_array(ints...));
	}

	template<class is_deriv>
	Shape(const Inner_Shape<is_deriv> param) {
		init(param);
	}
	template<int dim, class int_t>
	Shape (BC::array<dim, int_t> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		init(param);
	}
	__BCinline__ const auto inner_shape() const { return m_inner_shape; }
	__BCinline__ const auto outer_shape() const { return m_outer_shape; }
	__BCinline__ const auto block_shape() const { return outer_shape(); }

	__BCinline__ int size() const { return m_outer_shape[dims - 1]; }
	__BCinline__ int rows() const { return m_inner_shape[0]; }
	__BCinline__ int cols() const { return m_inner_shape[1]; }
	__BCinline__ int dimension(int i) const { return m_inner_shape[i]; }
	__BCinline__ int outer_dimension() const { return m_inner_shape[dims - 2]; }
	__BCinline__ int leading_dimension(int i) const { return m_outer_shape[i]; }
	__BCinline__ int block_dimension(int i) const  { return leading_dimension(i); }

protected:

	template<class T>
	void copy_shape(const Shape_Base<T>& shape) {
		for (int i = 0; i < dims; ++i) {
			m_inner_shape[i] = shape.dimension(i);
			m_outer_shape[i] = shape.block_dimension(i);
		}
	}
	void swap_shape(Shape& b) {
		std::swap(m_inner_shape, b.m_inner_shape);
		std::swap(m_outer_shape, b.m_outer_shape);
	}

private:

	template<class shape_t> __BCinline__
	void init(const shape_t& param) {
		m_inner_shape[0] = param[0];
		m_outer_shape[0] = m_inner_shape[0];
		for (int i = 1; i < dims; ++i) {
			m_inner_shape[i] = param[i];
			m_outer_shape[i] = m_outer_shape[i - 1] * m_inner_shape[i];
		}
	}

};

template<>
struct Shape<0> {

	__BCinline__ const auto inner_shape() const { return l_array<0>([&](auto x) { return 1; });}
	__BCinline__ const auto outer_shape() const { return l_array<0>([&](auto x) { return 0; });}
	__BCinline__ const auto block_shape() const { return l_array<0>([&](auto x) { return 1; });}
	__BCinline__ int size() const { return 1; }
	__BCinline__ int rows() const { return 1; }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return 1; }
	__BCinline__ int outer_dimension() const { return 1; }
	__BCinline__ int leading_dimension(int i) const { return 0; }
	__BCinline__ int block_dimension(int i) const { return 1; }

	template<class deriv> void copy_shape(const Shape_Base<deriv>& shape) {}
	static void swap_shape(Shape& a, Shape& b) {}

};

template<>
struct Shape<1> {

	BC::array<1, int> m_inner_shape;

	Shape (BC::array<1, int> param) : m_inner_shape(param) {}

	template<int dim, class f, class int_t>
	Shape (lambda_array<dim, int_t, f> param) {
		static_assert(dim >= 1, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
		m_inner_shape[0] = param[0];
	}

	Shape(int length_) : m_inner_shape(length_) {}
	__BCinline__ int size() const { return m_inner_shape[0]; }
	__BCinline__ int rows() const { return m_inner_shape[0]; }
	__BCinline__ int cols() const { return 1; }
	__BCinline__ int dimension(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
	__BCinline__ int outer_dimension() const { return 1; }
	__BCinline__ int leading_dimension(int i) const { return i == 0 ? 1 : i == 1 ? m_inner_shape[0] : 0; }
	__BCinline__ int block_dimension(int i)       const { return leading_dimension(i); }
	__BCinline__ const auto inner_shape() const { return m_inner_shape; }
	__BCinline__ const auto outer_shape() const { return l_array<1>([&](auto x) { return this->leading_dimension(x);});}
	__BCinline__ const auto block_shape() const { return m_inner_shape; }

	void copy_shape(const Shape<1>& shape) {
		this->m_inner_shape = shape.m_inner_shape;
	}

	template<class deriv> void copy_shape(const Shape_Base<deriv>& shape) {
		this->m_inner_shape[0] = shape.dimension(0);
	}

	void swap_shape(Shape<1>& shape){
		std::swap(m_inner_shape, shape.m_inner_shape);
	}


};



}



#endif /* SHAPE_H_ */
