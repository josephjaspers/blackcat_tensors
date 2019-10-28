/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_BASE_H_
#define BLACKCAT_TENSOR_BASE_H_

#include "Tensor_Common.h"
#include "Tensor_Operations.h"
#include "Tensor_Accessor.h"
#include "Tensor_IterAlgos.h"
#include "Tensor_Utility.h"

namespace BC {
namespace tensors {


template<class ExpressionTemplate>
class Tensor_Base :
		public ExpressionTemplate,
		public Tensor_Operations<ExpressionTemplate>,
		public Tensor_Utility<ExpressionTemplate>,
		public Tensor_Accessor<ExpressionTemplate>,
		public Tensor_IterAlgos<ExpressionTemplate> {

	template<class>
	friend class Tensor_Base;

	using self_type = Tensor_Base<ExpressionTemplate>;
	using parent = ExpressionTemplate;
	using operations = Tensor_Operations<ExpressionTemplate>;
	using accessor = Tensor_Accessor<ExpressionTemplate>;

public:

	using ExpressionTemplate::ExpressionTemplate;
	using ExpressionTemplate::internal;

	static constexpr int tensor_dimension =
			ExpressionTemplate::tensor_dimension;

	static constexpr int tensor_iterator_dimension =
			ExpressionTemplate::tensor_iterator_dimension;

	using value_type  = typename ExpressionTemplate::value_type;
	using system_tag  = typename ExpressionTemplate::system_tag;
	using traits_type = exprs::expression_traits<ExpressionTemplate>;

	using move_constructible = typename traits_type::is_move_constructible;
	using copy_constructible = typename traits_type::is_move_constructible;
	using move_assignable = typename traits_type::is_move_assignable;
	using copy_assignable = typename traits_type::is_copy_assignable;

	using operations::operator=;
	using operations::operator+;
	using operations::operator-;
	using operations::operator/;
	using operations::operator*;
	using operations::operator%;
	using operations::operator+=;
	using operations::operator-=;
	using operations::operator/=;
	using operations::operator%=;
	using operations::operator>;
	using operations::operator<;
	using operations::operator>=;
	using operations::operator<=;
	using operations::operator==;

	using accessor::operator[];
	using accessor::operator();

	Tensor_Base() = default;
	Tensor_Base(const parent&  param): parent(param) {}
	Tensor_Base(parent&& param): parent(param) {}

	template<class U>
	Tensor_Base(const Tensor_Base<U>& tensor):
		parent(tensor.as_parent()) {}

	Tensor_Base(BC::traits::only_if<copy_constructible::value, const self_type&> tensor):
		parent(tensor.as_parent()) {}

	Tensor_Base(BC::traits::only_if<move_constructible::value, self_type&&> tensor):
		parent(std::move(tensor.as_parent())) {}

	Tensor_Base& operator =(BC::traits::only_if<move_assignable::value, self_type&&> tensor) noexcept {
		this->as_parent() = std::move(tensor.as_parent());
		return *this;
	}

	Tensor_Base& operator =(BC::traits::only_if<copy_assignable::value, const self_type&> tensor) {
		 operations::operator=(tensor);
		 return *this;
	}

	Tensor_Base(BC::traits::only_if<tensor_dimension==0, value_type> scalar) {
		static_assert(tensor_dimension == 0, "SCALAR_INITIALIZATION ONLY AVAILABLE TO SCALARS");
		this->fill(scalar);
	}

	~Tensor_Base() {
		this->deallocate();
	}

private:

	parent& as_parent() {
		return static_cast<parent&>(*this);
	}
	const parent& as_parent() const {
		return static_cast<const parent&>(*this);
	}
};

}
}

#endif /* TENSOR_BASE_H_ */
