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

//aliases
template<class... Ts>
auto index(Ts... ts) {
	return BC::dim(ts...);
}

template<class T, class Shape>
auto reshape(Tensor_Base<T>& tensor, Shape shape)
{
	static_assert(BC::tensors::exprs::expression_traits<T>::is_array::value &&
					T::tensor_iterator_dimension <= 1,
					"Reshape is only available to continuous tensors");
	auto reshaped_tensor =  make_tensor(exprs::make_view(tensor, shape));
	BC_ASSERT(reshaped_tensor.size() == tensor.size(), "Reshape requires same size");
	return reshaped_tensor;
}

template<class T, class Shape>
const auto reshape(const Tensor_Base<T>& tensor, Shape shape)
{
	static_assert(BC::tensors::exprs::expression_traits<T>::is_array::value &&
		T::tensor_iterator_dimension <= 1,
		"Reshape is only available to continuous tensors");
	auto reshaped_tensor =  make_tensor(exprs::make_view(tensor, shape));
	BC_ASSERT(reshaped_tensor.size() == tensor.size(), "Reshape requires same size");
	return reshaped_tensor;
}


template<class ExpressionTemplate, class voider=void>
class Tensor_Accessor {

	const auto& as_derived() const { return static_cast<const Tensor_Base<ExpressionTemplate>&>(*this); }
	      auto& as_derived()       { return static_cast<      Tensor_Base<ExpressionTemplate>&>(*this); }

public:

	const auto operator [] (BC::size_t i) const { return slice(i); }
	      auto operator [] (BC::size_t i)       { return slice(i); }

	//enables syntax: `tensor[{start, end}]`
	const auto operator [] (BC::Dim<2> range) const { return slice(range[0], range[1]); }
	      auto operator [] (BC::Dim<2> range)       { return slice(range[0], range[1]); }

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

	const auto row_range(int begin, int end) const
	{
		static_assert(ExpressionTemplate::tensor_dimension  == 2,
				"ROW_RANGE ONLY AVAILABLE TO MATRICES");

		BC_ASSERT(begin < end,
				"Row range, begin-range must be smaller then end-range");
		BC_ASSERT(begin >= 0 && begin < as_derived().rows(),
				"Row range, begin-range must be between 0 and rows()");
		BC_ASSERT(end   >= 0 && end   < as_derived().rows(),
				"Row range, end-range must be be between begin-range and rows()");

		return chunk(this->as_derived(), begin, 0)(end-begin, this->as_derived().cols());
	}

	auto row_range(int begin, int end)
	{
		using self = Tensor_Accessor<ExpressionTemplate>;
		return BC::traits::auto_remove_const(
				const_cast<const self&>(*this).row_range(begin, end));
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

//Disable accessors for expression types
//This specialization is just for the cppyy interpretor,
//Tensor_Base has... using accessor[] ||
#ifdef BC_CLING_JIT
template<class ExpressionTemplate>
class Tensor_Accessor<ExpressionTemplate,
std::enable_if_t<exprs::expression_traits<ExpressionTemplate>::is_expr::value ||
	ExpressionTemplate::tensor_dimension == 0>>	 {

	const auto& as_derived() const { return static_cast<const Tensor_Base<ExpressionTemplate>&>(*this); }
		  auto& as_derived()	   { return static_cast<	  Tensor_Base<ExpressionTemplate>&>(*this); }


public:
	const int operator [] (int i) const {
		throw 1;
	}

	template<class... args>
	const void operator () (args... i) const {
		throw 1;
	}
};
#endif

}//end of module name space
}//end of BC name space


#endif /* TENSOR_SHAPING_H_ */
