/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

	const auto operator [](bc::size_t i) const {
		return slice(i);
	}

	auto operator [](bc::size_t i) {
		return slice(i);
	}

	//enables syntax: `tensor[{start, end}]`
	const auto operator [](bc::Dim<2> range) const {
		return slice(range[0], range[1]);
	}

	auto operator [](bc::Dim<2> range) {
		return slice(range[0], range[1]);
	}

	const auto slice(bc::size_t i) const
	{
		BC_ASSERT_ASSIGNABLE("const auto slice(bc::size_t i) const");
		BC_ASSERT(i >= 0 && i < this->outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(*this, i));
	}

	auto slice(bc::size_t i)
	{
		BC_ASSERT_ASSIGNABLE("auto slice(bc::size_t i)");
		BC_ASSERT(i >= 0 && i < this->outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(*this, i));
	}

	const auto slice(bc::size_t from, bc::size_t to) const
	{
		BC_ASSERT_ASSIGNABLE("const auto slice(bc::size_t from, bc::size_t to) const");
		BC_ASSERT(from >= 0 && to <= this->outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= this->outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(*this, from, to));
	}

	auto slice(bc::size_t from, bc::size_t to)
	{
		BC_ASSERT_ASSIGNABLE("auto slice(bc::size_t from, bc::size_t to)");
		BC_ASSERT(from >= 0 && to <= this->outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to <= this->outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(*this, from, to));
	}

	const auto scalar(bc::size_t i) const
	{
		BC_ASSERT_ASSIGNABLE("const auto scalar(bc::size_t i) const");
		BC_ASSERT(i >= 0 && i < this->size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(*this, i));
	}

	auto scalar(bc::size_t i)
	{
		BC_ASSERT_ASSIGNABLE("auto scalar(bc::size_t i)");
		BC_ASSERT(i >= 0 && i < this->size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(*this, i));
	}

	const auto diagnol(bc::size_t index = 0) const
	{
		BC_ASSERT_ASSIGNABLE("const auto diagnol(bc::size_t index = 0) const");
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -this->rows() && index < this->rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(*this,index));
	}

	auto diagnol(bc::size_t index = 0)
	{
		BC_ASSERT_ASSIGNABLE("auto diagnol(bc::size_t index = 0)");
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"diagnol method is only available to matrices");
		BC_ASSERT(index > -this->rows() && index < this->rows(),
				"diagnol `index` must be -rows() and rows())");
		return make_tensor(exprs::make_diagnol(*this,index));
	}

	//returns a copy of the tensor without actually copying the elements
	auto shallow_copy() const
	{
		BC_ASSERT_ASSIGNABLE("auto shallow_copy() const");
		return make_tensor(
				exprs::make_view(*this, this->get_shape()));
	}

	auto shallow_copy()
	{
		BC_ASSERT_ASSIGNABLE("auto shallow_copy()");
		return make_tensor(
				exprs::make_view(*this, this->get_shape()));
	}

	const auto col(bc::size_t i) const
	{
		BC_ASSERT_ASSIGNABLE("const auto col(bc::size_t i) const");
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto col(bc::size_t i)
	{
		BC_ASSERT_ASSIGNABLE("auto col(bc::size_t i)");
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	const auto row(bc::size_t index) const
	{
		BC_ASSERT_ASSIGNABLE("const auto row(bc::size_t index) const");
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < this->rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(*this, index));
	}

	auto row(bc::size_t index)
	{
		BC_ASSERT_ASSIGNABLE("const auto row(bc::size_t index) const");
		static_assert(ExpressionTemplate::tensor_dimension == 2,
				"MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		BC_ASSERT(index >= 0 && index < this->rows(),
				"Row index must be between 0 and rows()");
		return make_tensor(exprs::make_row(*this, index));
	}

private:
	using subblock_index  = bc::Dim<ExpressionTemplate::tensor_dimension>;
	using subblock_shape = bc::Shape<ExpressionTemplate::tensor_dimension>;
	using subblock_index_shape = std::tuple<subblock_index, subblock_shape>;
public:

	const auto subblock(subblock_index index, subblock_shape shape) const
	{
		BC_ASSERT_ASSIGNABLE("const auto subblock(subblock_index index, subblock_shape shape) const");
		return make_tensor(exprs::make_chunk(*this, index, shape));
	}

	auto subblock(subblock_index index, subblock_shape shape)
	{
		BC_ASSERT_ASSIGNABLE("auto subblock(subblock_index index, subblock_shape shape)");
		return make_tensor(exprs::make_chunk(*this, index, shape));
	}

	const auto operator [] (subblock_index_shape index_shape) const {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	auto operator [] (subblock_index_shape index_shape) {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	const auto operator() (bc::size_t i) const { return scalar(i); }
		  auto operator() (bc::size_t i)       { return scalar(i); }


	template<int X>
	auto reshaped(bc::Dim<X> shape)
	{
		BC_ASSERT_ASSIGNABLE("auto reshaped(bc::Dim<X> shape)");
		static_assert(ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == this->size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(*this, shape));
	}

	template<int X>
	const auto reshaped(bc::Dim<X> shape) const
	{
		BC_ASSERT_ASSIGNABLE("const auto reshaped(bc::Dim<X> shape) const");
		static_assert(ExpressionTemplate::tensor_iterator_dimension <= 1,
			"Reshape is only available to continuous tensors");

		BC_ASSERT(shape.size() == this->size(),
				"Reshape requires the new and old shape be same sizes");

		return make_tensor(exprs::make_view(*this, shape));
	}

	template<class... Integers>
	const auto reshaped(Integers... ints) const {
		return reshaped(bc::dim(ints...));
	}

	template<class... Integers>
	auto reshaped(Integers... ints) {
		return reshaped(bc::dim(ints...));
	}

	auto flattened() {
		return this->reshaped(this->size());
	}

	const auto flattened() const {
		return this->reshaped(this->size());
	}
