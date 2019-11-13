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
struct Array:
			private Allocator,
			public Kernel_Array<
					Shape,
					Scalar,
					typename BC::allocator_traits<Allocator>::system_tag,
					Tags...> {


	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using allocator_type = Allocator;
	using shape_type = Shape;
	using value_type = Scalar;
	using stream_type = Stream<system_tag>;

private:

	using parent_type = Kernel_Array<Shape, Scalar, system_tag, Tags...>;
	using allocator_traits_t = BC::allocator_traits<allocator_type>;

	stream_type m_stream;

public:

	const stream_type& get_stream() const { return m_stream; }
		  stream_type& get_stream()       { return m_stream; }

	Allocator get_allocator() const {
		return static_cast<const Allocator&>(*this);
	}

	Array() {
		if (Shape::tensor_dimension == 0) {
			this->m_data = get_allocator().allocate(1);
		}
	}

	Array(const Array& array):
		Allocator(allocator_traits_t::
			select_on_container_copy_construction(array)),
		parent_type(array.get_shape(), get_allocator()),
		m_stream(array.get_stream())
	{
		evaluate(
				make_bin_expr<BC::oper::Assign>(
						this->internal(),
						array.internal()), get_stream());
	}

	Array(Array&& array):
		Allocator(array.get_allocator()),
		parent_type(array),
			m_stream(array.get_stream())
	{
		array.m_data = nullptr;
	}

	//Construct via shape-like object and Allocator
	template<
		class ShapeLike,
		class=std::enable_if_t<
			!expression_traits<ShapeLike>::is_array::value &&
			!expression_traits<ShapeLike>::is_expr::value &&
			ShapeLike::tensor_dimension == parent_type::tensor_dimension &&
			Shape::tensor_dimension != 0>>
	Array(ShapeLike param, Allocator allocator=Allocator()):
		Allocator(allocator),
		parent_type(typename parent_type::shape_type(param), get_allocator()) {}

	//Constructor for integer sequence, IE Matrix(m, n)
	template<
		class... ShapeDims,
		class=std::enable_if_t<
			traits::sequence_of_v<BC::size_t, ShapeDims...> &&
			sizeof...(ShapeDims) == Shape::tensor_dimension>
	>
	Array(const ShapeDims&... shape_dims):
		parent_type(typename parent_type::shape_type(shape_dims...), get_allocator()) {}

	//Shape-like object with maybe allocator
	template<
		class Expression,
		class=std::enable_if_t<
			expression_traits<Expression>::is_array::value ||
			expression_traits<Expression>::is_expr::value>>
	Array(const Expression& expression, Allocator allocator=Allocator()):
		Allocator(allocator),
		parent_type(
			typename parent_type::shape_type(expression.inner_shape()),
			get_allocator())
	{
		evaluate(
				make_bin_expr<BC::oper::Assign>(
						this->internal(),
						expression.internal()), get_stream());
	}


	//If Copy-constructing from a slice, attempt to query the allocator
	//Restrict to same value_type (obviously), same dimensions (for fast-copy)
	template<class AltShape, class... SliceTags>
	Array(const Array_Slice<
			AltShape,
			value_type,
			allocator_type,
			SliceTags...>& expression):
		Allocator(allocator_traits_t::
			select_on_container_copy_construction(expression.get_allocator())),

		parent_type(typename parent_type::shape_type(
				expression.inner_shape()),
				get_allocator()),

		m_stream(expression.get_stream())
	{
		evaluate(
				make_bin_expr<BC::oper::Assign>(
						this->internal(),
						expression.internal()), get_stream());
	}

public:
	Array& operator = (Array&& array) {
		if (allocator_traits_t::is_always_equal::value ||
			array.get_allocator() == this->get_allocator())
		{
			std::swap((shape_type&)(*this), (shape_type&)array);
			std::swap(this->m_data, array.m_data);

			if (allocator_traits_t::
					propagate_on_container_move_assignment::value) {
				static_cast<Allocator&>(*this) =
						static_cast<Allocator&&>(array);
			}
		} else {
			get_allocator().deallocate(this->data(), this->size());
			(shape_type&)(*this) = (shape_type&)array;
			this->m_data = get_allocator().allocate(this->size());
			evaluate(
					make_bin_expr<BC::oper::Assign>(
							this->internal(),
							array.internal()), get_stream());
		}
		return *this;
	}

	void deallocate() {
		if (this->m_data) {
			Allocator::deallocate(this->data(), this->size());
			this->m_data= nullptr;
		}
	}

};


template<class Shape, class Allocator>
auto make_tensor_array(Shape shape, Allocator alloc) {
	using value_type = typename Allocator::value_type;
	return Array<Shape, value_type, Allocator>(shape, alloc);
}


} //ns BC
} //ns exprs
} //ns tensors



#endif /* SHAPE_H_ */
