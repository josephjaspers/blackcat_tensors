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
	return BC::make_dim(ts...);
}
template<int Dimension>
using index_type = BC::Dim<Dimension>;


template<class T, class Shape>
auto reshape(Tensor_Base<T>& tensor, Shape shape) {
	static_assert(BC::tensors::exprs::expression_traits<T>::is_array::value &&
					T::tensor_iterator_dimension <= 1,
					"Reshape is only available to continuous tensors");
	auto reshaped_tensor =  make_tensor(exprs::make_view(tensor, shape));
	BC_ASSERT(reshaped_tensor.size() == tensor.size(), "Reshape requires same size");
	return reshaped_tensor;
}


template<class T, class Shape>
const auto reshape(const Tensor_Base<T>& tensor, Shape shape) {
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
		  auto& as_derived()	   { return static_cast<	  Tensor_Base<ExpressionTemplate>&>(*this); }

public:

	auto data() const { return this->as_derived().memptr(); }
	auto data()	   { return this->as_derived().memptr(); }

	const auto operator [] (BC::size_t i) const { return slice(i); }
		  auto operator [] (BC::size_t i)	   { return slice(i); }

	struct range { BC::size_t  from, to; };	//enables syntax: `tensor[{start, end}]`
	const auto operator [] (range r) const { return slice(r.from, r.to); }
		  auto operator [] (range r)	   { return slice(r.from, r.to); }

	const auto subblock(
				index_type<ExpressionTemplate::tensor_dimension> index,
				BC::Shape<ExpressionTemplate::tensor_dimension> shape) const {
		return make_tensor(exprs::make_chunk(as_derived(), index, shape));

	}

	auto subblock(
			index_type<ExpressionTemplate::tensor_dimension> index,
			BC::Shape<ExpressionTemplate::tensor_dimension> shape) {
		return make_tensor(exprs::make_chunk(as_derived(), index, shape));
	}

private:
	using subblock_index_type = std::tuple<
			index_type<ExpressionTemplate::tensor_dimension>,
			BC::Shape<ExpressionTemplate::tensor_dimension>>;
public:

	const auto operator [] (subblock_index_type index_shape) const {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	auto operator [] (subblock_index_type index_shape) {
		return subblock(std::get<0>(index_shape), std::get<1>(index_shape));
	}

	const auto row_range(int begin, int end) const {
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
	auto row_range(int begin, int end) {
		return BC::traits::auto_remove_const(
				const_cast<const Tensor_Accessor<ExpressionTemplate>&>(*this).row_range(begin, end));
	}

	const auto scalar(BC::size_t i) const {
		BC_ASSERT(i >= 0 && i < as_derived().size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(as_derived(), i));
	}
	auto scalar(BC::size_t i) {
		BC_ASSERT(i >= 0 && i < as_derived().size(),
				"Scalar index must be between 0 and size()");
		return make_tensor(exprs::make_scalar(as_derived(), i));
	}

	const auto slice(BC::size_t i) const {
		BC_ASSERT(i >= 0 && i < as_derived().outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(as_derived(), i));
	}

	auto slice(BC::size_t i) {
		BC_ASSERT(i >= 0 && i < as_derived().outer_dimension(),
				"slice index must be between 0 and outer_dimension()");
		return make_tensor(exprs::make_slice(as_derived(), i));
	}

	const auto slice(BC::size_t from, BC::size_t to) const {
		BC_ASSERT(from >= 0 && to < as_derived().outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to < as_derived().outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(as_derived(), from, to));
	}

	auto slice(BC::size_t from, BC::size_t to) {
		BC_ASSERT(from >= 0 && to < as_derived().outer_dimension(),
				"slice `from` must be between 0 and outer_dimension()");
		BC_ASSERT(to > from && to < as_derived().outer_dimension(),
				"slice `to` must be between `from` and outer_dimension()");
		return make_tensor(exprs::make_ranged_slice(as_derived(), from, to));
	}

	const auto diagnol(BC::size_t index = 0) const {
		BC_ASSERT(index > -as_derived().rows() && index < as_derived().rows(),
				"diagnol `index` must be -rows() and rows())");
		static_assert(ExpressionTemplate::tensor_dimension  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
		return make_tensor(exprs::make_diagnol(as_derived(),index));
	}

	auto diagnol(BC::size_t index = 0) {
		BC_ASSERT(index > -as_derived().rows() && index < as_derived().rows(),
				"diagnol `index` must be -rows() and rows())");
		static_assert(ExpressionTemplate::tensor_dimension  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
		return make_tensor(exprs::make_diagnol(as_derived(),index));
	}

	//returns a copy of the tensor without actually copying the elements
	auto shallow_copy() const {
		return make_tensor(exprs::make_view(as_derived(), as_derived().get_shape()));
	}

	const auto col(BC::size_t i) const {
		static_assert(ExpressionTemplate::tensor_dimension == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	auto col(BC::size_t i) {
		static_assert(ExpressionTemplate::tensor_dimension == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	const auto row(BC::size_t index) const {
		BC_ASSERT(index >= 0 && index < as_derived().rows(),
				"Row index must be between 0 and rows()");
		static_assert(ExpressionTemplate::tensor_dimension == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(exprs::make_row(as_derived(), index));
	}

	auto row(BC::size_t index) {
		BC_ASSERT(index >= 0 && index < as_derived().rows(),
				"Row index must be between 0 and rows()");

		static_assert(ExpressionTemplate::tensor_dimension == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(exprs::make_row(as_derived(), index));
	}

	const auto operator() (BC::size_t i) const { return scalar(i); }
		  auto operator() (BC::size_t i)	   { return scalar(i); }

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
