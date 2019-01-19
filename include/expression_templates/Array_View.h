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
namespace et {


template<int Dimension, class Scalar, class Allocator>
struct ArrayViewExpr
        : Array_Base<ArrayViewExpr<Dimension, Scalar, Allocator>, Dimension>,
          Shape<Dimension> {

    using value_type = Scalar;
    using allocator_t = Allocator;
    using system_tag = typename BC::allocator_traits<allocator_t>::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = true;
    static constexpr int DIMS = Dimension;
    static constexpr int ITERATOR = DIMS;

    const value_type* array = nullptr;

    ArrayViewExpr()                      = default;
    ArrayViewExpr(const ArrayViewExpr& ) = default;
    ArrayViewExpr(      ArrayViewExpr&&) = default;

    __BCinline__
    const value_type* memptr() const  { return array; }

    void deallocate() {}

};

template<int Dimension, class Scalar, class Allocator>
struct Array_View : ArrayViewExpr<Dimension, Scalar, Allocator> {
	using parent = ArrayViewExpr<Dimension, Scalar, Allocator>;
	using parent::parent;


	const Allocator* alloc = nullptr;
	bool owns_allocator = false;


	Array_View() = default;
	Array_View(const Array_View& av) {
		copy_construct(av);
	}

	void copy_construct(const Array_View& view) {
		this->copy_shape(view);
		this->array = view.array;
	}

	void internal_swap(Array_View& swap) {
		std::swap(this->array, swap.array);
		this->swap_shape(swap);
	}

	void move_construct(Array_View& array_move) {
		this->internal_swap(array_move);
	}

	template<
		class tensor_t,
		typename = std::enable_if_t<
			tensor_t::DIMS == Dimension &&
			BC::is_array<tensor_t>()>
	>
	Array_View(const tensor_t& tensor) {
		this->array = tensor.memptr();
		this->copy_shape(tensor);
		this->alloc = &(tensor.get_allocator_ref());
	}

	Allocator& get_allocator_ref() {
		return *alloc;
	}
	const Allocator& get_allocator_ref() const {
		return *alloc;
	}

	auto& internal_base() { return *this; }
	const auto& internal_base() const { return *this; }

	void deallocate() {
		if (owns_allocator) {
			delete alloc;
		}
	}
};

}
}

#endif /* ARRAY_VIEW_H_ */
