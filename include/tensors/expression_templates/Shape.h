/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef INTERNAL_SHAPE_H_
#define INTERNAL_SHAPE_H_

namespace BC {
namespace tensors {
namespace exprs { 


template<int Dimension>
struct Shape {

	static constexpr int tensor_dimension = Dimension;
	static_assert(Dimension >= 0, "BC: SHAPE OBJECT MUST HAVE AT LEAST 0 OR MORE DIMENSIONS");

    BC::utility::array<Dimension, int> m_inner_shape = {0};
    BC::utility::array<Dimension, int> m_block_shape = {0};

    BCINLINE Shape() {}

    template<class... integers,
    class=std::enable_if_t<BC::traits::sequence_of_v<BC::size_t, integers...> && (sizeof...(integers) == Dimension)>>
    Shape(integers... ints) {
        static_assert(traits::sequence_of_v<int, integers...>, "INTEGER LIST OF SHAPE");
        static_assert(sizeof...(integers) == Dimension, "integer initialization must have the same number of dimensions");
        init(BC::utility::make_array(ints...));
    }

    template<int x, class=std::enable_if_t<(x >= Dimension)>> BCINLINE
    Shape(const Shape<x>& shape) {
        for (int i = 0; i < Dimension; ++i) {
            m_inner_shape[i] = shape.m_inner_shape[i];
            m_block_shape[i] = shape.m_block_shape[i];
        }
    }

    BCINLINE Shape(const BC::utility::array<Dimension, BC::size_t>& new_shape,
			const Shape<Dimension>& parent_shape) {
		m_inner_shape = new_shape;
		for (int i = 0; i < Dimension; ++i ) {
			m_block_shape[i] = parent_shape.leading_dimension(i);
		}
        calculate_size();
	}

    BCINLINE Shape(Shape<Dimension> new_shape, Shape<Dimension> parent_shape):
    	m_inner_shape(new_shape.m_inner_shape),
    	m_block_shape(parent_shape.m_block_shape) {
        calculate_size();
    }

    template<int dim, class int_t, class=std::enable_if_t<(dim>=Dimension)>>
    BCINLINE Shape (BC::utility::array<dim, int_t> param) {
        static_assert(dim >= Dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    template<int dim, class Function, class IntType, class=std::enable_if_t<(dim >= Dimension)>>
    BCINLINE Shape (utility::lambda_array<dim, IntType, Function> param) {
        static_assert(dim >= Dimension, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    BCINLINE const auto& inner_shape() const { return m_inner_shape; }
    BCINLINE const auto& outer_shape() const { return m_block_shape; }

    BCINLINE BC::size_t  size() const { return m_block_shape[Dimension - 1]; }
    BCINLINE BC::size_t  rows() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  cols() const { return m_inner_shape[1]; }
    BCINLINE BC::size_t  dimension(int i) const { return m_inner_shape[i]; }
    BCINLINE BC::size_t  outer_dimension() const { return m_inner_shape[Dimension - 1]; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return i < Dimension ? m_block_shape[i] : 0; }

    template<class... integers, typename=std::enable_if_t<BC::traits::sequence_of_v<BC::size_t, integers...>>>
    BCINLINE BC::size_t dims_to_index(integers... ints) const {
        return dims_to_index(BC::utility::make_array(ints...));
    }

    BCINLINE BC::size_t coefficientwise_dims_to_index(BC::size_t index) const {
        return index;
    }

    template<int D> BCINLINE
    BC::size_t dims_to_index(const BC::utility::array<D, int>& var) const {
        BC::size_t index = var[D-1];
        for(int i = 1; i < Dimension; ++i) {
            index += leading_dimension(i - 1) * var[D-1-i];
        }
        return index;
    }


private:

    template<class shape_t> BCINLINE
    void init(const shape_t& param) {
        m_inner_shape[0] = param[0];
        m_block_shape[0] = m_inner_shape[0];
        for (int i = 1; i < Dimension; ++i) {
            m_inner_shape[i] = param[i];
            m_block_shape[i] = m_block_shape[i - 1] * m_inner_shape[i];
        }

        calculate_size();
    }

    BCINLINE
    void calculate_size() {       //calculate size
		m_block_shape[Dimension-1] = m_inner_shape[0];
		for(int i = 1; i < Dimension; ++i) {
			m_block_shape[Dimension-1] *= m_inner_shape[i];
		}
    }

};

template<>
struct Shape<0> {

	static constexpr int tensor_dimension = 0;

    template<int x> BCINLINE
    Shape(const Shape<x>&) {} //empty

    BCINLINE Shape<0>() {}

    template<class... Args>
    BCINLINE Shape<0>(const Args&...) {}

    BCINLINE const auto inner_shape() const { return utility::make_lambda_array<0>([&](auto x) { return 1; });}
    BCINLINE const auto outer_shape() const { return utility::make_lambda_array<0>([&](auto x) { return 0; });}
    BCINLINE BC::size_t  size() const { return 1; }
    BCINLINE BC::size_t  rows() const { return 1; }
    BCINLINE BC::size_t  cols() const { return 1; }
    BCINLINE BC::size_t  dimension(int i) const { return 1; }
    BCINLINE BC::size_t  outer_dimension() const { return 1; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return 0; }

    template<class... integers>
    BCINLINE BC::size_t dims_to_index(integers... ints) const {
    	return 0;
    }

    BCINLINE BC::size_t coefficientwise_dims_to_index(BC::size_t i) const {
        return 0;
    }
};

template<>
struct Shape<1> {

	static constexpr int tensor_dimension = 1;

    BC::utility::array<1, int> m_inner_shape = {0};

    BCINLINE Shape() {};
    BCINLINE Shape (BC::utility::array<1, int> param) : m_inner_shape {param} {}

    template<int dim, class f, class int_t> BCINLINE
    Shape (utility::lambda_array<dim, int_t, f> param) {
        static_assert(dim >= 1, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        m_inner_shape[0] = param[0];
    }

    template<int x>
    BCINLINE Shape(const Shape<x>& shape) {
        static_assert(x >= 1, "BC: CANNOT CONSTRUCT A VECTOR SHAPE FROM A SCALAR SHAPE");
        m_inner_shape[0] = shape.m_inner_shape[0];
    }

    BCINLINE Shape(int length_) : m_inner_shape { length_ } {}
    BCINLINE BC::size_t  size() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  rows() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  cols() const { return 1; }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
    BCINLINE BC::size_t  outer_dimension() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return i == 0 ? 1 : 0; }
    BCINLINE const auto& inner_shape() const { return m_inner_shape; }
    BCINLINE auto outer_shape() const { return utility::make_lambda_array<0>([](int i) { return i == 0 ? 1 : 0; }); }

    template<class... integers>
    BCINLINE BC::size_t dims_to_index(BC::size_t i, integers... ints) const {
    	return dims_to_index(ints...);
    }
    template<class... integers>
    BCINLINE BC::size_t dims_to_index(BC::size_t i) const {
    	return i;
    }

    BCINLINE BC::size_t coefficientwise_dims_to_index(BC::size_t i) const {
        return i;
    }
};


struct Strided_Vector_Shape {

	static constexpr int tensor_dimension = 1;

    BC::utility::array<1, int> m_inner_shape = {0};
    BC::utility::array<1, int> m_block_shape = {1};

    BCINLINE Strided_Vector_Shape(BC::size_t length, BC::size_t leading_dimension) {
        m_inner_shape[0] = length;
        m_block_shape[0] = leading_dimension;
    }

    BCINLINE BC::size_t  size() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  rows() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  cols() const { return 1; }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
    BCINLINE BC::size_t  outer_dimension() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return i == 0 ? m_block_shape[0] : 0; }
    BCINLINE const auto& inner_shape() const { return m_inner_shape; }
    BCINLINE const auto& outer_shape() const { return m_block_shape; }

    template<class... integers>
    BCINLINE BC::size_t dims_to_index(BC::size_t i, integers... ints) const {
    	return dims_to_index(ints...);
    }

    template<class... integers>
    BCINLINE BC::size_t dims_to_index(BC::size_t i) const {
    	return m_block_shape[0] * i;
    }

    template<class... integers>
    BCINLINE BC::size_t slice_ptr_index(BC::size_t i) const {
    	return m_block_shape[0] * i;
    }

    BCINLINE BC::size_t coefficientwise_dims_to_index(BC::size_t index) const {
        return m_block_shape[0] * index;
    }
};

template<class... integers, typename=std::enable_if_t<traits::sequence_of_v<BC::size_t, integers...>>>
auto shape(integers... ints) {
	return Shape<sizeof...(integers)>(ints...);
}

template<class InnerShape, typename=std::enable_if_t<!traits::sequence_of_v<BC::size_t, InnerShape>>>
auto shape(InnerShape is) {
	return Shape<InnerShape::tensor_dimension>(is);
}

}
}

using BC::tensors::exprs::Shape;
using BC::tensors::exprs::shape;

} //ns BC


#endif /* SHAPE_H_ */
