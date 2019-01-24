/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef INTERNAL_SHAPE_H_
#define INTERNAL_SHAPE_H_

#include "Common.h"
#include "Shape_Base.h"


namespace BC {
namespace et {


template<int dims, class derived=void>
struct Shape : Shape_Base<std::conditional_t<std::is_void<derived>::value, Shape<dims, derived>, derived>> {

	static_assert(dims >= 0, "BC: SHAPE OBJECT MUST HAVE AT LEAST 0 OR MORE DIMENSIONS");
	using self = std::conditional_t<std::is_void<derived>::value, Shape<dims, derived>, derived>;

    BC::array<dims, int> m_inner_shape = {0};
    BC::array<dims, int> m_block_shape = {0};

    __BCinline__ Shape() {}

    template<class... integers>
    Shape(integers... ints) {
        static_assert(MTF::seq_of<int, integers...>, "INTEGER LIST OF SHAPE");
        static_assert(sizeof...(integers) == dims, "integer initialization must have the same number of dimensions");
        init(BC::make_array(ints...));
    }

    template<int x> __BCinline__
    Shape(const Shape<x>& shape) {
        static_assert(x >= dims, "Shape Construction internal error");
        for (int i = 0; i < dims; ++i) {
            m_inner_shape[i] = shape.m_inner_shape[i];
            m_block_shape[i] = shape.m_block_shape[i];
        }
    }

    template<int dim, class int_t>
    __BCinline__ Shape (BC::array<dim, int_t> param) {
        static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    template<int dim, class f, class int_t>
    __BCinline__ Shape (lambda_array<dim, int_t, f> param) {
        static_assert(dim >= dims, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        init(param);
    }
    __BCinline__ const auto& inner_shape() const { return m_inner_shape; }
    __BCinline__ const auto& outer_shape() const { return m_block_shape; }
    __BCinline__ const auto& block_shape() const { return outer_shape(); }

    __BCinline__ BC::size_t  size() const { return m_block_shape[dims - 1]; }
    __BCinline__ BC::size_t  rows() const { return m_inner_shape[0]; }
    __BCinline__ BC::size_t  cols() const { return m_inner_shape[1]; }
    __BCinline__ BC::size_t  dimension(int i) const { return m_inner_shape[i]; }
    __BCinline__ BC::size_t  outer_dimension() const { return m_inner_shape[dims - 2]; }
    __BCinline__ BC::size_t  leading_dimension(int i) const { return i < dims ? m_block_shape[i] : 0; }
    __BCinline__ BC::size_t  block_dimension(int i) const  { return leading_dimension(i); }

protected:

    template<class T>
    void copy_shape(const Shape_Base<T>& shape) {
        for (int i = 0; i < dims; ++i) {
            m_inner_shape[i] = shape.dimension(i);
            m_block_shape[i] = shape.block_dimension(i);
        }
    }
    void swap_shape(Shape& b) {
        std::swap(m_inner_shape, b.m_inner_shape);
        std::swap(m_block_shape, b.m_block_shape);
    }

private:

    template<class shape_t> __BCinline__
    void init(const shape_t& param) {
        m_inner_shape[0] = param[0];
        m_block_shape[0] = m_inner_shape[0];
        for (int i = 1; i < dims; ++i) {
            m_inner_shape[i] = param[i];
            m_block_shape[i] = m_block_shape[i - 1] * m_inner_shape[i];
        }
    }

};

template<>
struct Shape<0> {

    template<int x> __BCinline__
    Shape(const Shape<x>&) {} //empty

    __BCinline__ Shape<0>() {}
    __BCinline__ const auto inner_shape() const { return l_array<0>([&](auto x) { return 1; });}
    __BCinline__ const auto outer_shape() const { return l_array<0>([&](auto x) { return 0; });}
    __BCinline__ const auto block_shape() const { return l_array<0>([&](auto x) { return 1; });}
    __BCinline__ BC::size_t  size() const { return 1; }
    __BCinline__ BC::size_t  rows() const { return 1; }
    __BCinline__ BC::size_t  cols() const { return 1; }
    __BCinline__ BC::size_t  dimension(int i) const { return 1; }
    __BCinline__ BC::size_t  outer_dimension() const { return 1; }
    __BCinline__ BC::size_t  leading_dimension(int i) const { return 0; }
    __BCinline__ BC::size_t  block_dimension(int i) const { return 1; }

    template<class deriv> void copy_shape(const Shape_Base<deriv>& shape) {}
    static void swap_shape(Shape& a) {}
};

template<>
struct Shape<1> {

    BC::array<1, int> m_inner_shape = {0};
    BC::array<1, int> m_block_shape = {1};

    __BCinline__ Shape() {};
    __BCinline__ Shape (BC::array<1, int> param) : m_inner_shape {param}, m_block_shape { 1 }  {}

    template<int dim, class f, class int_t> __BCinline__
    Shape (lambda_array<dim, int_t, f> param) {
        static_assert(dim >= 1, "SHAPE MUST BE CONSTRUCTED FROM ARRAY OF AT LEAST SAME dimension");
        m_inner_shape[0] = param[0];
        m_block_shape[0] = 1;
    }

    template<int x, class der>
    __BCinline__ Shape(const Shape<x, der>& shape) {
        static_assert(x >= 1, "BC: CANNOT CONSTRUCT A VECTOR SHAPE FROM A SCALAR SHAPE");
        m_inner_shape[0] = shape.m_inner_shape[0];
        m_block_shape[0] = 1; //shape.m_block_shape[0];
    }

    __BCinline__ Shape(int length, BC::size_t  leading_dimension) {
        m_inner_shape[0] = length;
        m_block_shape[0] = leading_dimension;
    }

    __BCinline__ Shape(int length_) : m_inner_shape { length_ }, m_block_shape {1} {}
    __BCinline__ BC::size_t  size() const { return m_inner_shape[0]; }
    __BCinline__ BC::size_t  rows() const { return m_inner_shape[0]; }
    __BCinline__ BC::size_t  cols() const { return 1; }
    __BCinline__ BC::size_t  dimension(int i) const { return i == 0 ? m_inner_shape[0] : 1; }
    __BCinline__ BC::size_t  outer_dimension() const { return m_inner_shape[0]; }
    __BCinline__ BC::size_t  leading_dimension(int i) const { return i == 0 ? m_block_shape[0] : 0; }
    __BCinline__ BC::size_t  block_dimension(int i)   const { return leading_dimension(i); }
    __BCinline__ const auto& inner_shape() const { return m_inner_shape; }
    __BCinline__ const auto& outer_shape() const { return m_block_shape; }
    __BCinline__ const auto& block_shape() const { return m_inner_shape; }

    void copy_shape(const Shape<1>& shape) {
        this->m_inner_shape = shape.m_inner_shape;
        this->m_block_shape = shape.m_block_shape;
    }

    template<class deriv> void copy_shape(const Shape_Base<deriv>& shape) {
        this->m_inner_shape[0] = shape.dimension(0);
        this->m_block_shape[0] = shape.dimension(0);
    }

    void swap_shape(Shape<1>& shape){
        std::swap(m_inner_shape, shape.m_inner_shape);
        std::swap(m_block_shape, shape.m_block_shape);
    }


};

template<int ndims>
struct SubShape : Shape<ndims, SubShape<ndims>> {

	using parent = Shape<ndims, SubShape<ndims>>;
	BC::array<ndims, BC::size_t> m_outer_shape;

	SubShape() = default;


	template<class der>
	SubShape(const BC::array<ndims, BC::size_t>& new_shape, const Shape<ndims, der>& parent_shape)
	: parent(new_shape) {
		for (int i = 0; i < ndims; ++i ) {
			m_outer_shape[i] = parent_shape.leading_dimension(i);
		}
	}
	SubShape(const BC::array<ndims, BC::size_t>& new_shape, const SubShape<ndims>& parent_shape)
	: parent(new_shape) {
		for (int i = 0; i < ndims; ++i ) {
			m_outer_shape[i] = parent_shape.m_outer_shape[i];
		}
	}


    __BCinline__ const auto& outer_shape() const { return m_outer_shape; }
    __BCinline__ BC::size_t  leading_dimension(int i) const { return m_outer_shape[i]; }



private:
	//hide from external sources
	using Shape<ndims, SubShape<ndims>>::swap_shape;
	using Shape<ndims, SubShape<ndims>>::copy_shape;
};

}

//push shape into BC namespace
template<int x>
using Shape = et::Shape<x>;

template<class... integers, typename=std::enable_if_t<MTF::seq_of<BC::size_t, integers...>>>
auto make_shape(integers... ints) {
	return Shape<sizeof...(integers)>(ints...);
}
}



#endif /* SHAPE_H_ */
