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


template<int dims>
struct Shape {

	static_assert(dims >= 0, "BC: SHAPE OBJECT MUST HAVE AT LEAST 0 OR MORE DIMENSIONS");

    BC::utility::array<dims, int> m_inner_shape = {0};
    BC::utility::array<dims, int> m_block_shape = {0};

    BCINLINE Shape() {}

    template<class... integers,
    class=std::enable_if_t<BC::traits::sequence_of_v<BC::size_t, integers...> && (sizeof...(integers) == dims)>>
    Shape(integers... ints) {
        static_assert(traits::sequence_of_v<int, integers...>, "INTEGER LIST OF SHAPE");
        static_assert(sizeof...(integers) == dims, "integer initialization must have the same number of dimensions");
        init(BC::utility::make_array(ints...));
    }

    template<int x> BCINLINE
    Shape(const Shape<x>& shape) {
        static_assert(x >= dims, "Shape Construction internal error");
        for (int i = 0; i < dims; ++i) {
            m_inner_shape[i] = shape.m_inner_shape[i];
            m_block_shape[i] = shape.m_block_shape[i];
        }
    }


    BCINLINE
	Shape(
			const BC::utility::array<dims, BC::size_t>& new_shape,
			const Shape<dims>& parent_shape) {
		m_inner_shape = new_shape;
		for (int i = 0; i < dims; ++i ) {
			m_block_shape[i] = parent_shape.leading_dimension(i);
		}
        calculate_size();
	}

    template<int dim, class int_t, class=std::enable_if_t<(dim>=dims)>>
    BCINLINE Shape (BC::utility::array<dim, int_t> param) {
        static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    template<int dim, class Function, class IntType, class=std::enable_if_t<(dim >= dims)>>
    BCINLINE Shape (utility::lambda_array<dim, IntType, Function> param) {
        static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    BCINLINE const auto& inner_shape() const { return m_inner_shape; }
    BCINLINE const auto& outer_shape() const { return m_block_shape; }

    BCINLINE BC::size_t  size() const { return m_block_shape[dims - 1]; }
    BCINLINE BC::size_t  rows() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  cols() const { return m_inner_shape[1]; }
    BCINLINE BC::size_t  dimension(int i) const { return m_inner_shape[i]; }
    BCINLINE BC::size_t  outer_dimension() const { return m_inner_shape[dims - 1]; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return i < dims ? m_block_shape[i] : 0; }

    template<class... integers, typename=std::enable_if_t<BC::traits::sequence_of_v<BC::size_t, integers...>>>
    BCINLINE BC::size_t dims_to_index(integers... ints) const {
        return dims_to_index(BC::utility::make_array(ints...));
    }

    template<int D> BCINLINE
    BC::size_t dims_to_index(const BC::utility::array<D, int>& var) const {
        BC::size_t  index = var[0];
        for(int i = 1; i < dims; ++i) {
            index += leading_dimension(i - 1) * var[i];
        }
        return index;
    }


private:

    template<class shape_t> BCINLINE
    void init(const shape_t& param) {
        m_inner_shape[0] = param[0];
        m_block_shape[0] = m_inner_shape[0];
        for (int i = 1; i < dims; ++i) {
            m_inner_shape[i] = param[i];
            m_block_shape[i] = m_block_shape[i - 1] * m_inner_shape[i];
        }

        calculate_size();
    }

    BCINLINE
    void calculate_size() {       //calculate size
		m_block_shape[dims-1] = m_inner_shape[0];
		for(int i = 1; i < dims; ++i) {
			m_block_shape[dims-1] *= m_inner_shape[i];
		}
    }

};

template<>
struct Shape<0> {

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
};

template<>
struct Shape<1> {

    BC::utility::array<1, int> m_inner_shape = {0};
    BC::utility::array<1, int> m_block_shape = {1};

    BCINLINE Shape() {};
    BCINLINE Shape (BC::utility::array<1, int> param) : m_inner_shape {param}, m_block_shape { 1 }  {}

    template<int dim, class f, class int_t> BCINLINE
    Shape (utility::lambda_array<dim, int_t, f> param) {
        static_assert(dim >= 1, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        m_inner_shape[0] = param[0];
        m_block_shape[0] = 1;
    }

    template<int x>
    BCINLINE Shape(const Shape<x>& shape) {
        static_assert(x >= 1, "BC: CANNOT CONSTRUCT A VECTOR SHAPE FROM A SCALAR SHAPE");
        m_inner_shape[0] = shape.m_inner_shape[0];
        m_block_shape[0] = 1; //shape.m_block_shape[0];
    }

    BCINLINE Shape(BC::size_t length, BC::size_t leading_dimension) {
        m_inner_shape[0] = length;
        m_block_shape[0] = leading_dimension;
    }

    BCINLINE Shape(int length_) : m_inner_shape { length_ }, m_block_shape {1} {}
    BCINLINE BC::size_t  size() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  rows() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  cols() const { return 1; }
    BCINLINE BC::size_t  dimension(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
    BCINLINE BC::size_t  outer_dimension() const { return m_inner_shape[0]; }
    BCINLINE BC::size_t  leading_dimension(int i) const { return i == 0 ? m_block_shape[0] : 0; }
    BCINLINE const auto& inner_shape() const { return m_inner_shape; }
    BCINLINE const auto& outer_shape() const { return m_block_shape; }

    template<class... integers>
    BCINLINE constexpr BC::size_t dims_to_index(BC::size_t i, integers... ints) const {
    	return m_block_shape[0] * i;
    }

};

} //ns exprs
} //ns tensors

//push shape into BC namespace
template<int x>
using Shape = tensors::exprs::Shape<x>;

template<class... integers, typename=std::enable_if_t<traits::sequence_of_v<BC::size_t, integers...>>>
auto make_shape(integers... ints) {
	return Shape<sizeof...(integers)>(ints...);
}

template<class InnerShape, typename=std::enable_if_t<!traits::sequence_of_v<BC::size_t, InnerShape>>>
auto make_shape(InnerShape is) {
	return Shape<InnerShape::tensor_dimension>(is);
}

} //ns BC


#endif /* SHAPE_H_ */
