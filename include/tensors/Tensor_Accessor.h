/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BC_ASSERT_VIEWABLE\
		static_assert(\
				exprs::expression_traits<ExpressionTemplate>::is_array::value,\
				"Views are only available to Memory owning types");

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
		BC_ASSERT_VIEWABLE
		BC_ASSERT(i >= 0 && i < this->outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(*this, i));
	}

	auto slice(BC::size_t i)
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT(i >= 0 && i < this->outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(*this, i));
	}

	const auto slice(BC::size_t from, BC::size_t to) const
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT(from >= 0 && to <= this->outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= this->outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(*this, from, to));
	}

	auto slice(BC::size_t from, BC::size_t to)
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT(from >= 0 && to <= this->outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= this->outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(*this, from, to));
	}

	const auto scalar(BC::size_t i) const
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT(i >= 0 && i < this->size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(*this, i));
	}

	auto scalar(BC::size_t i)
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT(i >= 0 && i < this->size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(*this, i));
	}

	const auto diagnol(BC::size_t index = 0) const
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -this->rows() && index < this->rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(*this,index));
	}

	auto diagnol(BC::size_t index = 0)
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -this->rows() && index < this->rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(*this,index));
	}

	//returns a copy of the tensor without actually copying the elements
	auto shallow_copy() const
	{
		BC_ASSERT_VIEWABLE
		return make_tensor(
				exprs::make_view(*this, this->get_shape()));
	}

	auto shallow_copy()
	{
		BC_ASSERT_VIEWABLE
		BC_ASSERT_VIEWABLE
		return make_tensor(
				exprs::make_view(*this, this->get_shape()));
	}

	const auto col(BC::size_t i) const
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto col(BC::size_t i)
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	const auto row(BC::size_t index) const
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < this->rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(*this, index));
	}

	auto row(BC::size_t index)
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < this->rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(*this, index));
	}

private:
	using subblock_index  = BC::Dim<ExpressionTemplate::tensor_dimension>;
	using subblock_shape = BC::Shape<ExpressionTemplate::tensor_dimension>;
	using subblock_index_shape = std::tuple<subblock_index, subblock_shape>;
public:

	const auto subblock(subblock_index index, subblock_shape shape) const
	{
		BC_ASSERT_VIEWABLE
		return make_tensor(exprs::make_chunk(*this, index, shape));
	}

	auto subblock(subblock_index index, subblock_shape shape)
	{
		BC_ASSERT_VIEWABLE
		return make_tensor(exprs::make_chunk(*this, index, shape));
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
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == this->size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(*this, shape));
	}

	template<int X>
	const auto reshaped(BC::Dim<X> shape) const
	{
		BC_ASSERT_VIEWABLE
		static_assert(ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == this->size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(*this, shape));
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
		return this->reshaped(this->size());
	}

	const auto flattened() const {
		return this->reshaped(this->size());
	}
