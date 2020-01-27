/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNAL_BASE_H_
#define BC_INTERNAL_BASE_H_

#include "expression_template_traits.h"

namespace bc {
namespace tensors {
namespace exprs {

template<class Derived>
struct Expression_Template_Base  {

	BCINLINE
	const Derived& internal() const {
		return static_cast<const Derived&>(*this);
	}

	BCINLINE
	Derived& internal() {
		return static_cast<Derived&>(*this);
	}


	BCINLINE Expression_Template_Base() {

		using bc::traits::true_call;
		using bc::traits::Integer;

#ifndef _MSC_VER
		static_assert(std::is_trivially_copy_constructible<Derived>::value,
				"ExpressionTemplates must be trivially constructible");

		static_assert(std::is_trivially_copyable<Derived>::value,
				"ExpressionTemplates must be tricially copyable");
#endif
		static_assert(true_call<typename Derived::value_type>(),
				"ExpressionTemplates must define: 'using value_type = <T>;'");

		static_assert(true_call<decltype(std::declval<Derived>().inner_shape())>(),
				"ExpressionTemplates must define: inner_shape()");

		static_assert(true_call<decltype(std::declval<Derived>().rows())>(),
				"ExpressionTemplates must define: rows()");

		static_assert(true_call<decltype(std::declval<Derived>().cols())>(),
				"ExpressionTemplates must define: cols()");

		static_assert(true_call<decltype(std::declval<Derived>().dim(0))>(),
				"ExpressionTemplates must define: dim(int)");

		static_assert(true_call<Integer<Derived::tensor_dim>>(),
				"ExpressionTemplates must define: "
				"static constexpr int tensor_dim");

		static_assert(true_call<Integer<Derived::tensor_iterator_dim>>(),
				"ExpressionTemplates must define: "
				"static constexpr int tensor_iterator_dim");

		static_assert(Derived::tensor_iterator_dim >= Derived::tensor_dim
				|| Derived::tensor_iterator_dim <= 1,
				"Iterator Dimension must be greater than or equal to the tensor_dim");
	}

	void deallocate() const {}
};


template<class Derived>
struct Expression_Base:
		Expression_Template_Base<Derived>  {

	using copy_constructible = std::false_type;
	using move_constructible = std::false_type;
	using copy_assignable    = std::false_type;
	using move_assignable    = std::false_type;
	using expression_template_expression_type = std::true_type;

	BCINLINE const auto inner_shape() const {
		bc::Dim<Derived::tensor_dim> dim;
		for (bc::size_t i = 0; i < Derived::tensor_dim; ++i) {
			dim[i] = static_cast<const Derived&>(*this).dim(i);
		}
		return dim;
	}

	BCINLINE const auto get_shape() const {
		return bc::Shape<Derived::tensor_dim>(
				static_cast<const Derived&>(*this).inner_shape());
	}

	BCINLINE bc::size_t outer_dim() const {
		auto& derived = static_cast<const Derived&>(*this);
		return derived.dim(derived.tensor_dim-1);
	}
};


template<class Derived>
struct Kernel_Array_Base : Expression_Template_Base<Derived> {

	using expression_template_array_type = std::true_type;

	BCINLINE
	Kernel_Array_Base() {
		using value_type = typename Derived::value_type;
		using data_type = decltype(std::declval<Derived>().data());
		using data_value_type =
				std::remove_const_t<std::remove_pointer_t<data_type>>;

		static_assert(std::is_same<value_type, data_value_type>::value,
			"Array_Types must define: data() \n"
			"which returns a pointer of the ExpressionTemplate's value_type");
	}
};


} //ns BC
} //ns exprs
} //ns tensors

#endif /* BC_INTERNAL_BASE_H_ */
