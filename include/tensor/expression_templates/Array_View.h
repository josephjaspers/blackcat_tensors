/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_VIEW_H_

#include "Array_Base.h"
#include "Array.h"


namespace BC {
namespace exprs {
namespace array_view {

template<int Dimension, class Scalar, class SystemTag>
struct Const_View
        : Array_Base<Const_View<Dimension, Scalar, SystemTag>, Dimension>,
          Shape<Dimension> {

    using value_type = Scalar;
    using system_tag = SystemTag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;

    static constexpr int DIMS = Dimension;
    static constexpr int ITERATOR = DIMS;

    const value_type* array = nullptr;

    Const_View()                      = default;
    Const_View(const Const_View& ) = default;
    Const_View(      Const_View&&) = default;

    BCINLINE
    const value_type* memptr() const  { return array; }

	template<class Tensor>
	Const_View(Tensor& tensor, BC::exprs::Shape<Dimension> shape, int index)
	: BC::exprs::Shape<Dimension>(shape),
	  array(&tensor[index]) {}
};
}


template<int Dimension, class Scalar, class Allocator>
struct Array_Const_View
		: array_view::Const_View<Dimension,
		  	  	  	  	  	  	  Scalar,
		  	  	  	  	  	  	  typename BC::allocator_traits<Allocator>::system_tag> {

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using context_type = BC::Context<system_tag>;
	using parent =  array_view::Const_View<Dimension, Scalar, system_tag>;
	using parent::parent;

	context_type context;
	const Allocator* alloc = nullptr;

	Array_Const_View() = default;
	Array_Const_View(const Array_Const_View&) = default;
	Array_Const_View(Array_Const_View&&) = default;

	void internal_move(Array_Const_View& swap) {
		this->context = swap.context;
		this->alloc   = & swap.get_allocator();

		std::swap(this->array, swap.array);
		this->swap_shape(swap);
	}

	template<
		class tensor_t,
		typename = std::enable_if_t<
			tensor_t::DIMS == Dimension &&
			expression_traits<tensor_t>::is_array>
	>
	Array_Const_View(const tensor_t& tensor) {
		this->array = tensor.memptr();
		this->copy_shape(tensor);
		this->alloc = &(tensor.get_allocator());
		this->context = tensor.get_context();
	}

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
