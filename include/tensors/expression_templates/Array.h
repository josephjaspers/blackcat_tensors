/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_H_


#include "Expression_Template_Base.h"
#include "Array_Kernel_Array.h"

namespace BC {
namespace tensors {
namespace exprs {

template<class, class, class, class...>
class Array_Slice;


template<class Shape, class Scalar, class Allocator, class... Tags>
class Array :
			private Allocator,
			public Kernel_Array<Shape, Scalar, typename BC::allocator_traits<Allocator>::system_tag, Tags...> {

	using self = Array<Shape, Scalar, Allocator, Tags...>;
	using parent = Kernel_Array<Shape, Scalar,  typename BC::allocator_traits<Allocator>::system_tag, Tags...>;
	using stream_type = Stream<typename BC::allocator_traits<Allocator>::system_tag>;

public:

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using allocator_t = Allocator;
	using stream_t   = Stream<system_tag>;
	using value_type = Scalar;

private:

	stream_type m_stream;

public:

	const stream_t& get_stream()   const { return m_stream; }
	stream_t& get_stream()   { return m_stream; }
	Allocator get_allocator() const { return static_cast<const Allocator&>(*this); }

	Array() {
		if (Shape::tensor_dimension == 0) {
			this->memptr_ref() = get_allocator().allocate(1);
		}
	}

	Array(const Array& array)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(array)),
	  parent(array.get_shape(), get_allocator()),
	  m_stream(array.get_stream()) {
		greedy_evaluate(this->internal(), array.internal(), get_stream());
	}

	Array(Array&& array) //TODO handle propagate_on_container_move_assignment
	: Allocator(array.get_allocator()),
	  parent(array),
	  m_stream(array.get_stream()) {
		array.memptr_ref() = nullptr;
		//This causes segmentation fault with NVCC currently (compiler segfault, not runtime)
//		array_.get_shape_ref() = BC::Shape<Dimension>(); //resets the shape
	}

	//Construct via shape-like object and Allocator
    template<
    	class ShapeLike,
    	class=std::enable_if_t<
    		!expression_traits<ShapeLike>::is_array::value &&
    		!expression_traits<ShapeLike>::is_expr::value &&
    		Shape::tensor_dimension != 0>>
    Array(ShapeLike param, Allocator allocator=Allocator()):
    	Allocator(allocator),
    	parent(typename parent::shape_type(param), get_allocator()) {}

	//Constructor for integer sequence, IE Matrix(m, n)
	template<
		class... ShapeDims,
		class=std::enable_if_t<
			traits::sequence_of_v<BC::size_t, ShapeDims...> &&
			sizeof...(ShapeDims) == Shape::tensor_dimension>
	>
	Array(const ShapeDims&... shape_dims):
		parent(typename parent::shape_type(shape_dims...), get_allocator()) {}

	//Shape-like object with maybe allocator
    template<
    	class Expression,
    	class=std::enable_if_t<expression_traits<Expression>::is_array::value || expression_traits<Expression>::is_expr::value>>
	Array(const Expression& expression, Allocator allocator=Allocator()):
		Allocator(allocator),
		parent(typename parent::shape_type(expression.inner_shape()), get_allocator()) {
		greedy_evaluate(this->internal(), expression.internal(), get_stream());
	}


	//If Copy-constructing from a slice, attempt to query the allocator
    //Restrict to same value_type (obviously), same dimensions (for fast-copy)
    //And restrict to continuous (as we should attempt to support Sparse matrices in the future)
    template<class... SliceTags>
	Array(const Array_Slice<Shape, value_type, allocator_t, SliceTags...>& expression):
		Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(expression.get_allocator())),
		parent(typename parent::shape_type(expression.inner_shape()), get_allocator()),
		m_stream(expression.get_stream()) {
		greedy_evaluate(this->internal(), expression.internal(), this->get_stream());
	}

public:
    Array& operator = (Array&& array) {
    		if (BC::allocator_traits<Allocator>::is_always_equal::value ||
    			array.get_allocator() == this->get_allocator()) {
			std::swap(this->memptr_ref(), array.memptr_ref());
			this->get_shape_ref() = array.get_shape_ref();

	    	if (BC::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
				static_cast<Allocator&>(*this) = static_cast<Allocator&&>(array);
			}
    	} else {
    		get_allocator().deallocate(this->memptr_ref(), this->size());
			this->get_shape_ref() = array.get_shape_ref();
			this->memptr_ref() = get_allocator().allocate(this->size());
			greedy_evaluate(this->internal(), array.internal(), this->get_stream());
    	}
    		return *this;
    }

    void deallocate() {
       Allocator::deallocate(this->memptr_ref(), this->size());
       this->memptr_ref() = nullptr;
	}

};


template<class Shape, class Allocator>
auto make_tensor_array(Shape shape, Allocator alloc) {
	return Array<Shape, typename Allocator::value_type, Allocator>(shape, alloc);
}


} //ns BC
} //ns exprs
} //ns tensors



#endif /* SHAPE_H_ */
