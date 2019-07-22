/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_

#include "Expression_Template_Base.h"
#include "Shape.h"
#include "Array.h"


namespace BC {
namespace tensors {
namespace exprs {

template<int Dimension, class Scalar, class Allocator>
struct Array_Const_View
		: Kernel_Array<Dimension,
						  Scalar,
						  typename BC::allocator_traits<Allocator>::system_tag, BC_Noncontinuous> {

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using stream_type = BC::Stream<system_tag>;
	using parent =  Kernel_Array<Dimension, Scalar, system_tag, BC_Noncontinuous>;
	using allocator_t = Allocator;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;

    static constexpr int tensor_dimension = Dimension;
    static constexpr int tensor_iterator_dimension = tensor_dimension;

	stream_type stream;
	Allocator alloc;

	Array_Const_View() = default;
	Array_Const_View(const Array_Const_View&) = default;
	Array_Const_View(Array_Const_View&&) = default;

	Array_Const_View& operator = (Array_Const_View&& acv) {
		this->stream  = acv.stream;
		this->alloc   = acv.get_allocator();
		this->memptr_ref()    = acv.memptr();
		this->get_shape_ref() = acv.get_shape();
		return *this;
	}

	template<
		class tensor_t,
		typename = std::enable_if_t<
			tensor_t::tensor_dimension == Dimension &&
			expression_traits<tensor_t>::is_array>
	>
	Array_Const_View(const tensor_t& tensor)
	: parent(typename parent::shape_type(tensor.get_shape()), tensor.memptr()),
	  stream(tensor.get_stream()),
	  alloc(tensor.get_allocator()) {}

	template<
		class tensor_t,
		typename = std::enable_if_t<
			tensor_t::tensor_dimension == Dimension &&
			expression_traits<tensor_t>::is_array>
	>
	Array_Const_View(const tensor_t& tensor, allocator_t allocator)
	: parent(typename parent::shape_type(tensor.get_shape()), tensor.memptr()),
	  stream(tensor.get_stream()),
	  alloc(allocator) {}

	Allocator get_allocator() const {
		return alloc;
	}

	const stream_type& get_stream()   const { return stream; }
		  stream_type& get_stream()   		{ return stream; }

};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* ARRAY_VIEW_H_ */
