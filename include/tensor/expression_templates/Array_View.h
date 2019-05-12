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
namespace exprs {

template<int Dimension, class Scalar, class Allocator>
struct Array_Const_View
		: ArrayExpression<Dimension,
						  Scalar,
						  typename BC::allocator_traits<Allocator>::system_tag, BC_Noncontinuous> {

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using context_type = BC::Context<system_tag>;
	using parent =  ArrayExpression<Dimension, Scalar, system_tag, BC_Noncontinuous>;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;

    static constexpr int DIMS = Dimension;
    static constexpr int ITERATOR = DIMS;

	context_type context;
	const Allocator* alloc = nullptr;

	Array_Const_View() = default;
	Array_Const_View(const Array_Const_View&) = default;
	Array_Const_View(Array_Const_View&&) = default;

	void internal_move(Array_Const_View&& swap) {
		this->context = swap.context;
		this->alloc   = & swap.get_allocator();
		this->array   = swap.array;
		static_cast<SubShape<Dimension>&>(*this) = SubShape<Dimension>(swap);
	}

	template<
		class tensor_t,
		typename = std::enable_if_t<
			tensor_t::DIMS == Dimension &&
			expression_traits<tensor_t>::is_array>
	>
	Array_Const_View(const tensor_t& tensor)
	: parent(typename parent::shape_type(tensor.get_shape()), tensor.memptr()),
	  context(tensor.get_context()),
	  alloc(&tensor.get_allocator()) {}

	const Allocator& get_allocator() {
		return *alloc;
	}
	const Allocator& get_allocator() const {
		return *alloc;
	}

	const auto& internal_base() const { return *this; }
		  auto& internal_base()       { return *this; }
};
}
}

#endif /* ARRAY_VIEW_H_ */
