/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

#include "Tensor_Common.h"
#include "Expression_Templates/Array_Slice.h"
#include "Expression_Templates/Array_Slice_Range.h"
#include "Expression_Templates/Array_Scalar.h"
#include "Expression_Templates/Array_Chunk.h"
#include "Expression_Templates/Array_Reshape.h"
#include "Expression_Templates/Array_Format.h"
#include "Expression_Templates/Array_Strided_Vector.h"

namespace BC {
namespace module {

template<class derived>
struct Tensor_Shaping {

	static constexpr int DIMS() { return derived::DIMS(); }

private:

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived()       { return static_cast<	  derived&>(*this); }

	const auto& internal() const { return as_derived().internal(); }
		  auto& internal()       { return as_derived().internal(); }

public:

	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	struct range { int from, to; };
	const auto operator [] (range r) const { return slice(r.from, r.to); }
		  auto operator [] (range r) 	   { return slice(r.from, r.to); }

	const auto scalar(int i) const { return make_tensor(internal::make_scalar(internal(), i)); }
		  auto scalar(int i) 	   { return make_tensor(internal::make_scalar(internal(), i)); }

	const auto slice(int i) const  { return make_tensor(internal::make_slice(internal(), i)); }
		  auto slice(int i) 	   { return make_tensor(internal::make_slice(internal(), i)); }

	const auto slice(int from, int to) const  { return make_tensor(internal::make_ranged_slice(internal(),from,  to)); }
		  auto slice(int from, int to) 	   	  { return make_tensor(internal::make_ranged_slice(internal(),from,  to)); }


	const auto diag(int index = 0) const {
		static_assert(derived::DIMS()  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
		return make_tensor(internal::make_diagnol(internal(),index));
	}
	auto diag(int index = 0) {
		static_assert(derived::DIMS()  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
		return make_tensor(internal::make_diagnol(internal(),index));
	}

	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	const auto row(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(internal::make_row(internal(), i));
	}
	auto row(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return make_tensor(internal::make_row(internal(), i));
	}

	const auto operator() (int i) const { return scalar(i); }
		  auto operator() (int i) 	    { return scalar(i); }

};

}


template<class T>
const auto reshape(const Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(internal::make_reshape(tensor.internal(), BC::make_array(integers...)));
	};
}
template<class T>
auto reshape(Tensor_Base<T>& tensor) {
	return [&](auto... integers) {
		return make_tensor(internal::make_reshape(tensor.internal(), BC::make_array(integers...)));
	};
}

template<class T, class... integers>// class enabler = std::enable_if_t<MTF::seq_of<int, integers...>>>
const auto chunk(const Tensor_Base<T>& tensor, integers... ints) {
	return [&, ints...](auto... shape_indicies) {
		return make_tensor(internal::make_chunk(
				tensor.internal(),
				BC::make_array(ints...),
				BC::make_array(shape_indicies...)));
	};
}

template<class T, class... integers>// class enabler = std::enable_if_t<MTF::seq_of<int, integers...>>>
auto chunk(Tensor_Base<T>& tensor, integers... ints) {
	return [&, ints...](auto... shape_indicies) {
		return make_tensor(internal::make_chunk(
				tensor.internal(),
				BC::make_array(ints...),
				BC::make_array(shape_indicies...)));
	};
}

template<class T, class... integers>
auto format_as(Tensor_Base<T>& tensor,  int zero, integers... ints) {
	return make_tensor(internal::make_format(tensor.internal(), BC::make_array(ints...)));
}
template<class T, class... integers>
const auto format_as(const Tensor_Base<T>& tensor, int zero, integers... ints) {
	return make_tensor(internal::make_format(tensor.internal(), BC::make_array(ints...)));
}

}
#endif /* TENSOR_SHAPING_H_ */
