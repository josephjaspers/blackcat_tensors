/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_ACCESSOR_H_
#define BLACKCAT_TENSOR_ACCESSOR_H_

namespace BC {
namespace tensors {

template<class>
class Tensor_Base;

template<class ExpressionTemplate, class voider=void>
class Tensor_Accessor {

	const auto& as_derived() const {
		return static_cast<const Tensor_Base<ExpressionTemplate>&>(*this);
	}

	auto& as_derived() {
		return static_cast<Tensor_Base<ExpressionTemplate>&>(*this);
	}

public:

	const auto operator [](BC::size_t i) const {
		return slice(i);
	}

	auto operator [](BC::size_t i) {
		return slice(i);
	}

	//enables syntax: `tensor[{start, end}]`
	const auto operator [](BC::Dim<2> range) const {
		return slice(range[0], range[1]);
	}

	auto operator [](BC::Dim<2> range) {
		return slice(range[0], range[1]);
	}

	const auto slice(BC::size_t i) const
	{
		BC_ASSERT(i >= 0 && i < as_derived().outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(as_derived(), i));
	}

	auto slice(BC::size_t i)
	{
		BC_ASSERT(i >= 0 && i < as_derived().outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(as_derived(), i));
	}

	const auto slice(BC::size_t from, BC::size_t to) const
	{
		BC_ASSERT(from >= 0 && to <= as_derived().outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= as_derived().outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(as_derived(), from, to));
	}

	auto slice(BC::size_t from, BC::size_t to)
	{
		BC_ASSERT(from >= 0 && to <= as_derived().outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= as_derived().outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(as_derived(), from, to));
	}

	const auto scalar(BC::size_t i) const
	{
		BC_ASSERT(i >= 0 && i < as_derived().size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(as_derived(), i));
	}

	auto scalar(BC::size_t i)
	{
		BC_ASSERT(i >= 0 && i < as_derived().size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(as_derived(), i));
	}

	const auto diagnol(BC::size_t index = 0) const
	{
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -as_derived().rows() && index < as_derived().rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(as_derived(),index));
	}

	auto diagnol(BC::size_t index = 0)
	{
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -as_derived().rows() && index < as_derived().rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(as_derived(),index));
	}

	//returns a copy of the tensor without actually copying the elements
	auto shallow_copy() const
	{
		return make_tensor(
				exprs::make_view(as_derived(), as_derived().get_shape()));
	}

	auto shallow_copy()
	{
		return make_tensor(
				exprs::make_view(as_derived(), as_derived().get_shape()));
	}

	const auto col(BC::size_t i) const
	{
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto col(BC::size_t i)
	{
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	const auto row(BC::size_t index) const
	{
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < as_derived().rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(as_derived(), index));
	}

	auto row(BC::size_t index)
	{
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < as_derived().rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(as_derived(), index));
	}

private:
	using subblock_index  = BC::Dim<ExpressionTemplate::tensor_dimension>;
	using subblock_shape = BC::Shape<ExpressionTemplate::tensor_dimension>;
	using subblock_index_shape = std::tuple<subblock_index, subblock_shape>;
public:

	const auto subblock(subblock_index index, subblock_shape shape) const {
		return make_tensor(exprs::make_chunk(as_derived(), index, shape));
	}

	auto subblock(subblock_index index, subblock_shape shape) {
		return make_tensor(exprs::make_chunk(as_derived(), index, shape));
	}

	const auto operator [] (subblock_index_shape index_shape) const {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	auto operator [] (subblock_index_shape index_shape) {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	const auto operator() (BC::size_t i) const { return scalar(i); }
		  auto operator() (BC::size_t i)       { return scalar(i); }


	template<int X>
	auto reshaped(BC::Dim<X> shape)
	{
		static_assert(
				exprs::expression_traits<ExpressionTemplate>::is_array::value &&
						ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == as_derived().size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(as_derived(), shape));
	}

	template<int X>
	const auto reshaped(BC::Dim<X> shape) const
	{
		static_assert(
				exprs::expression_traits<ExpressionTemplate>::is_array::value &&
						ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == as_derived().size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(as_derived(), shape));
	}

	template<class... Integers>
	const auto reshaped(Integers... ints) const {
		return reshaped(BC::dim(ints...));
	}

	template<class... Integers>
	auto reshaped(Integers... ints) {
		return reshaped(BC::dim(ints...));
	}

	auto flattened() {
		return this->reshaped(this->as_derived().size());
	}

	const auto flattened() const {
		return this->reshaped(this->as_derived().size());
	}
};

template<class T, class Shape>
[[deprecated]] auto reshape(Tensor_Base<T>& tensor, Shape shape)
{
	static_assert(std::is_void<T>::value, "reshape has been deprecated, please use: `tensor.reshaped(ints...)` or `tensor.reshaped(Dim<x>)`");
}

template<class T, class Shape>
[[deprecated]] const auto reshape(const Tensor_Base<T>& tensor, Shape shape)
{
	static_assert(std::is_void<T>::value, "reshape has been deprecated, please use: `tensor.reshaped(ints...)` or `tensor.reshaped(Dim<x>)`;");
}

}//end of module name space
}//end of BC name space


#endif /* TENSOR_SHAPING_H_ */
