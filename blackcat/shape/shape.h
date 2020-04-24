/*
 * Shape.h
 *
 *  Created on: Sep 24, 2019
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_SHAPE_SHAPE_H_
#define BLACKCATTENSORS_SHAPE_SHAPE_H_

#include "dim.h"

namespace bc {


template<int N>
struct Shape {

	static_assert(N >= 0, "Shape<N>: ASSERT 'N >= 0'");

	template<int>
	friend struct Shape;

	static constexpr int tensor_dim = N;
	using size_t = bc::size_t;
	using value_type = size_t;

private:

	Dim<N> m_inner_shape = {0};
	Dim<N> m_block_shape = {0};

public:

	BCINLINE Shape() {};

	template<
		class... Integers,
		class=std::enable_if_t<
			bc::traits::sequence_of_v<size_t, Integers...> &&
			(sizeof...(Integers) == N)>> BCINLINE
	Shape(Integers... ints):
		m_inner_shape {ints...} {

		m_block_shape[0] = 1;
		for (int i = 1; i < N; ++i)
			m_block_shape[i] = m_inner_shape[i-1] * m_block_shape[i-1];
	}

	template<int X, class=std::enable_if_t<(X > N)>> BCINLINE
	Shape(const Shape<X>& shape):
		m_inner_shape(shape.m_inner_shape.template subdim<0, N>()),
		m_block_shape(shape.m_block_shape.template subdim<0, N>()) {}

	BCINLINE
	Shape(Dim<N> new_shape, const Shape<N>& parent_shape):
		m_inner_shape(new_shape),
		m_block_shape(parent_shape.m_block_shape) {}

	BCINLINE
	Shape(const Shape<N>& new_shape, const Shape<N>& parent_shape):
		m_inner_shape(new_shape.m_inner_shape),
		m_block_shape(parent_shape.m_block_shape) {
	}

	BCINLINE
	Shape(Dim<N> dims):
		m_inner_shape(dims.template subdim<0, N>()) {

		m_block_shape[0] = 1;
		for (int i = 1; i < N; ++i) {
			m_block_shape[i] = m_inner_shape[i-1] * m_block_shape[i-1];
		}
	}

	BCINLINE const auto& inner_shape() const { return m_inner_shape; }
	BCINLINE const auto& outer_shape() const { return m_block_shape; }
	BCINLINE size_t operator [] (size_t i) const { return m_inner_shape[i]; }
	BCINLINE size_t size() const { return m_inner_shape.size(); }
	BCINLINE size_t rows() const { return m_inner_shape[0]; }
	BCINLINE size_t cols() const { return m_inner_shape[1]; }
	BCINLINE size_t dim(int i) const { return m_inner_shape.dim(i); }
	BCINLINE size_t outer_dim() const { return m_inner_shape.outer_dim(); }

	BCINLINE size_t leading_dim(int i=N-1) const {
		return i < N ? m_block_shape[i] : 0;
	}

	BCINLINE bool operator == (const Shape& other) const {
		return m_inner_shape ==other.m_inner_shape;
	}

	BCINLINE size_t coefficientwise_dims_to_index(size_t index) const {
		return index;
	}

	template<
		class... Integers,
		class=std::enable_if_t<
			bc::traits::sequence_of_v<size_t, Integers...> &&
			(sizeof...(Integers) >= N)>> BCINLINE
	size_t dims_to_index(Integers... ints) const {
		return dims_to_index(bc::dim(ints...));
	}

	template<int D, class=std::enable_if_t<(D>=N)>> BCINLINE
	size_t dims_to_index(const Dim<D>& var) const {
		size_t index = var[D-1];
		for(int i = 1; i < N; ++i) {
			index += leading_dim(i) * var[D-1-i];
		}
		return index;
	}
};

template<>
struct Shape<0> {

	using size_t = bc::size_t;
	using value_type = size_t;

	static constexpr int tensor_dim = 0;

	template<int>
	friend struct Shape;

	static constexpr Dim<0> m_inner_shape = {};

	BCINLINE Shape<0>() {}

	template<class... Args>
	BCINLINE Shape<0>(const Args&...) {}

	BCINLINE Dim<0> inner_shape() const { return m_inner_shape; }
	BCINLINE size_t operator [] (size_t i) { return 1; }
	BCINLINE size_t size() const { return 1; }
	BCINLINE size_t rows() const { return 1; }
	BCINLINE size_t cols() const { return 1; }
	BCINLINE size_t dim(int i) const { return 1; }
	BCINLINE size_t outer_dim() const { return 1; }
	BCINLINE size_t leading_dim(int i=0) const { return 0; }
	BCINLINE bool operator == (const Shape& other) const { return true; }

	BCINLINE
	size_t coefficientwise_dims_to_index(size_t i) const {
		return 0;
	}

	template<class... Integers> BCINLINE
	size_t dims_to_index(Integers... ints) const {
		return 0;
	}
};

template<>
struct Shape<1> {

	using size_t = bc::size_t;
	using value_type = size_t;

	static constexpr int tensor_dim = 1;

	template<int>
	friend struct Shape;

private:

	bc::Dim<1> m_inner_shape = {0};

public:

	BCINLINE Shape() {};
	BCINLINE Shape (bc::Dim<1> param):
		m_inner_shape {param} {}

	template<int X, class=std::enable_if_t<(X>=1)>> BCINLINE
	Shape(const Shape<X>& shape) {
		m_inner_shape[0] = shape.m_inner_shape[0];
	}

	BCINLINE
	Shape(int length):
		m_inner_shape { length } {}

	BCINLINE size_t operator [] (size_t i) const { return dim(i); }
	BCINLINE size_t size() const { return m_inner_shape[0]; }
	BCINLINE size_t rows() const { return m_inner_shape[0]; }
	BCINLINE size_t cols() const { return 1; }
	BCINLINE size_t dim(size_t i) const { return i == 0 ? m_inner_shape[0] : 1; }
	BCINLINE size_t outer_dim() const { return m_inner_shape[0]; }
	BCINLINE size_t leading_dim(size_t i=0) const { return i == 0 ? 1 : 0; }
	BCINLINE const auto& inner_shape() const { return m_inner_shape; }

	BCINLINE bool operator == (const Shape<1>& other) const {
		return rows() == other.rows();
	}

	template<class... Integers> BCINLINE
	size_t dims_to_index(size_t i, Integers... ints) const {
		return dims_to_index(ints...);
	}

	template<class... Integers> BCINLINE
	size_t dims_to_index(size_t i) const {
		return i;
	}

	BCINLINE size_t coefficientwise_dims_to_index(size_t i) const {
		return i;
	}
};


struct Strided_Vector_Shape {

	static constexpr int tensor_dim = 1;

	bc::Dim<1> m_inner_shape = {0};
	bc::Dim<1> m_block_shape = {1};

	BCINLINE Strided_Vector_Shape(size_t length, size_t leading_dim) {
		m_inner_shape[0] = length;
		m_block_shape[0] = leading_dim;
	}

	BCINLINE bool operator == (const Strided_Vector_Shape& other) const {
		return rows() == other.rows();
	}

	BCINLINE friend bool operator == (
			const Strided_Vector_Shape& shape, const Shape<1>& other) {
		return shape.rows() == other.rows();
	}

	BCINLINE size_t operator [] (size_t idx) const { return dim(idx); }
	BCINLINE size_t size() const { return m_inner_shape[0]; }
	BCINLINE size_t rows() const { return m_inner_shape[0]; }
	BCINLINE size_t cols() const { return 1; }
	BCINLINE size_t dim(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
	BCINLINE size_t outer_dim() const { return m_inner_shape[0]; }
	BCINLINE size_t leading_dim(int i=0) const {
		return i == 0 ? m_block_shape[0] : 0;
	}

	BCINLINE const auto& inner_shape() const { return m_inner_shape; }

	template<class... Integers>
	BCINLINE size_t dims_to_index(size_t i, Integers... ints) const {
		return dims_to_index(ints...);
	}

	template<class... Integers>
	BCINLINE size_t dims_to_index(size_t index) const {
		return m_block_shape[0] * index;
	}

	BCINLINE size_t coefficientwise_dims_to_index(size_t index) const {
		return m_block_shape[0] * index;
	}
};

template<
		class... Integers,
		typename=std::enable_if_t<
				traits::sequence_of_v<size_t, Integers...>>>
BCINLINE auto shape(Integers... ints) {
	return Shape<sizeof...(Integers)>(ints...);
}

template<
		class InnerShape,
		typename=std::enable_if_t<!
				traits::sequence_of_v<size_t, InnerShape>>>
BCINLINE auto shape(InnerShape is) {
	return Shape<InnerShape::tensor_dim>(is);
}

}

#endif /* SHAPE_H_ */
