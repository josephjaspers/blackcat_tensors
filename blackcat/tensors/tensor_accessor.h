/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*
* Note const methods must come second to support cppyy (python bindings)
* Issue: https://bitbucket.org/wlav/cppyy/issues/224/overload-resolution-of-const-non-const
*/
#ifndef BLACKCAT_TENSORS_TENSOR_ACCESSOR_H_
#define BLACKCAT_TENSORS_TENSOR_ACCESSOR_H_

namespace bc {
namespace tensors {

template<class>
struct Tensor_Base;

template<int TensorDimension, class ExpressionTemplate>
class Tensor_Accessor_Base;

struct Scalar_Accessor
{
	void operator [] (int index) const {
		throw std::invalid_argument("[] not supported for scalars");
	}

	void operator () (int index) const {
		throw std::invalid_argument("() not supported for scalars");
	}
};

template<class ExpressionTemplate>
using Tensor_Accessor = std::conditional_t<
	ExpressionTemplate::tensor_dim == 0,
	Scalar_Accessor,
	Tensor_Accessor_Base<
		ExpressionTemplate::tensor_dim,
		Tensor_Base<ExpressionTemplate>>>;

template<int TensorDimension, class Derived>
class Tensor_Accessor_Base
{
	// static_assert(TensorDimension != 0);
	static constexpr int tensor_dim = TensorDimension;
	Derived& derived() { return static_cast<Derived&>(*this); }
	const Derived& derived() const { return static_cast<const Derived&>(*this); }
public:

	const auto operator [](bc::size_t i) const {
		return slice(i);
	}

	auto operator [](bc::size_t i) {
		return slice(i);
	}

	const auto operator [](bc::Dim<2> range) const {
		return slice(range[0], range[1]);
	}

	//enables syntax: `tensor[{start, end}]`
	auto operator [](bc::Dim<2> range) {
		return slice(range[0], range[1]);
	}

	auto slice(bc::size_t i) const {
		BC_ASSERT(i >= 0 && i < derived().outer_dim(),
			"slice index must be between 0 and outer_dim()");
		return make_tensor(exprs::make_slice(derived(), i));
	}

	auto slice(bc::size_t i)
	{
		BC_ASSERT(i >= 0 && i < derived().outer_dim(),
			"slice index must be between 0 and outer_dim()");
		return make_tensor(exprs::make_slice(derived(), i));
	}

	auto slice(bc::size_t from, bc::size_t to) const
	{
		BC_ASSERT(from >= 0 && to <= derived().outer_dim(),
			"slice `from` must be between 0 and outer_dim()");
		BC_ASSERT(to > from && to <= derived().outer_dim(),
			"slice `to` must be between `from` and outer_dim()");
		return make_tensor(exprs::make_ranged_slice(derived(), from, to));
	}

	auto slice(bc::size_t from, bc::size_t to)
	{
		BC_ASSERT(from >= 0 && to <= derived().outer_dim(),
			"slice `from` must be between 0 and outer_dim()");
		BC_ASSERT(to > from && to <= derived().outer_dim(),
			"slice `to` must be between `from` and outer_dim()");
		return make_tensor(exprs::make_ranged_slice(derived(), from, to));
	}

	auto scalar(bc::size_t i) const
	{
		BC_ASSERT(i >= 0 && i < derived().size(),
			"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(derived(), i));
	}

	auto scalar(bc::size_t i)
	{
		BC_ASSERT(i >= 0 && i < derived().size(),
			"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(derived(), i));
	}

	auto diagnol(bc::size_t index = 0) const
	{
		static_assert(tensor_dim  == 2,
			"diagnol method is only available to matrices");
		BC_ASSERT(index > -derived().rows() && index < derived().rows(),
			"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(derived(), index));
	}

	auto diagnol(bc::size_t index = 0)
	{
		static_assert(tensor_dim  == 2,
			"diagnol method is only available to matrices");
		BC_ASSERT(index > -derived().rows() && index < derived().rows(),
			"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(derived(), index));
	}

	auto shallow_copy() const
	{
		return make_tensor(exprs::make_view(derived(), derived().get_shape()));
	}

	//returns a copy of the tensor without actually copying the elements
	auto shallow_copy()
	{
		return make_tensor(exprs::make_view(derived(), derived().get_shape()));
	}

	auto col(bc::size_t i) const
	{
		static_assert(tensor_dim == 2,
			"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto col(bc::size_t i)
	{
		static_assert(tensor_dim == 2,
			"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto row(bc::size_t index) const
	{
		static_assert(tensor_dim == 2,
			"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < derived().rows(),
			"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(derived(), index));
	}

	auto row(bc::size_t index)
	{
		static_assert(tensor_dim == 2,
			"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < derived().rows(),
			"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(derived(), index));
	}

	auto subblock(Dim<tensor_dim> index, Dim<tensor_dim> shape) const
	{
		BC_ASSERT((index.reversed() + shape <= derived().inner_shape()).all(),
			"Index + Shape must be less parent shape");
		BC_ASSERT(((index>=0).all() && (shape>=0).all()),
			"Shape and Index must be greater than 0");
		return make_tensor(exprs::make_chunk(derived(), index, shape));
	}

	auto subblock(Dim<tensor_dim> index, Dim<tensor_dim> shape)
	{
		BC_ASSERT((index.reversed() + shape <= derived().inner_shape()).all(),
			"Index + Shape must be less parent shape");
		BC_ASSERT(((index>=0).all() && (shape>=0).all()),
			"Shape and Index must be greater than 0");
		return make_tensor(exprs::make_chunk(derived(), index, shape));
	}

	auto operator [] (
			std::tuple<Dim<tensor_dim>, Dim<tensor_dim>> index_shape) const {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	auto operator [] (
			std::tuple<Dim<tensor_dim>, Dim<tensor_dim>> index_shape) {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	      auto operator() (bc::size_t i)       { return scalar(i); }
	const auto operator() (bc::size_t i) const { return scalar(i); }

	template<int X>
	auto reshaped(bc::Dim<X> shape) const
	{
		static_assert(Derived::tensor_iterator_dim <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == derived().size(),
			"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(derived(), shape));
	}

	template<int X>
	auto reshaped(bc::Dim<X> shape)
	{
		static_assert(Derived::tensor_iterator_dim <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == derived().size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(derived(), shape));
	}

	template<class... Integers>
	auto reshaped(Integers... ints) const {
		return reshaped(bc::dim(ints...));
	}

	template<class... Integers>
	auto reshaped(Integers... ints) {
		return reshaped(bc::dim(ints...));
	}

	const auto flattened() const {
		return this->reshaped(derived().size());
	}

	auto flattened() {
		return this->reshaped(derived().size());
	}
};

}
}

#endif
