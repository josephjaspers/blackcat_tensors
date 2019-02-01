/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_TENSOR_ACCESSOR_H_
#define BLACKCAT_TENSOR_ACCESSOR_H_


#include "expression_templates/Array.h"
#include "expression_templates/Array_Slice.h"
#include "expression_templates/Array_Strided_Vector.h"


namespace BC {
namespace module {

template<class derived>
class Tensor_Accessor {

    const auto& as_derived() const { return static_cast<const derived&>(*this); }
          auto& as_derived()       { return static_cast<      derived&>(*this); }

    const auto& internal() const { return as_derived().internal_base(); }
          auto& internal()       { return as_derived().internal_base(); }

public:

    auto data() const { return this->as_derived().memptr(); }
    auto data()       { return this->as_derived().memptr(); }

    const auto operator [] (BC::size_t i) const { return slice(i); }
          auto operator [] (BC::size_t i)       { return slice(i); }

    struct range { BC::size_t  from, to; };	//enables syntax: `tensor[{start, end}]`
    const auto operator [] (range r) const { return slice(r.from, r.to); }
          auto operator [] (range r)       { return slice(r.from, r.to); }

    const auto scalar(BC::size_t i) const { return make_tensor(et::make_scalar(internal(), i)); }
          auto scalar(BC::size_t i)       { return make_tensor(et::make_scalar(internal(), i)); }

	const auto slice(BC::size_t i) const {
		return make_tensor(et::make_slice(internal(), i));
	}

	auto slice(BC::size_t i) {
		return make_tensor(et::make_slice(internal(), i));
	}

	const auto slice(BC::size_t from, BC::size_t to) const {
		return make_tensor(et::make_ranged_slice(internal(), from, to));
	}

	auto slice(BC::size_t from, BC::size_t to) {
		return make_tensor(et::make_ranged_slice(internal(), from, to));
	}


    const auto diag(BC::size_t index = 0) const {
        static_assert(derived::DIMS  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
        return make_tensor(et::make_diagnol(internal(),index));
    }

    auto diag(BC::size_t index = 0) {
        static_assert(derived::DIMS  == 2, "DIAGNOL ONLY AVAILABLE TO MATRICES");
        return make_tensor(et::make_diagnol(internal(),index));
    }

    const auto col(BC::size_t i) const {
        static_assert(derived::DIMS == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
        return slice(i);
    }

    auto col(BC::size_t i) {
        static_assert(derived::DIMS == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
        return slice(i);
    }

    const auto row(BC::size_t i) const {
        static_assert(derived::DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
        return make_tensor(et::make_row(internal(), i));
    }

    auto row(BC::size_t i) {
        static_assert(derived::DIMS == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
        return make_tensor(et::make_row(internal(), i));
    }

    const auto operator() (BC::size_t i) const { return scalar(i); }
          auto operator() (BC::size_t i)       { return scalar(i); }

};

}//end of module name space


template<class T>
const auto reshape(const Tensor_Base<T>& tensor) {
    return [&](auto... integers) {
        return make_tensor(et::make_view(tensor.internal_base(), BC::make_array(integers...)));
    };
}
template<class T>
auto reshape(Tensor_Base<T>& tensor) {
    return [&](auto... integers) {
        return make_tensor(et::make_view(tensor.internal_base(), BC::make_array(integers...)));
    };
}

template<class T, class... integers, class enabler = std::enable_if_t<meta::seq_of<BC::size_t, integers...>>>
const auto chunk(const Tensor_Base<T>& tensor, integers... ints) {
	auto index_point =  BC::make_array(ints...);

    return [&, index_point](auto... shape_indicies) {
        return make_tensor(et::make_chunk(
                tensor.internal_base(),
                index_point,
                BC::make_array(shape_indicies...)));
    };
}

template<class T, class... integers, class enabler = std::enable_if_t<meta::seq_of<BC::size_t, integers...>>>
auto chunk(Tensor_Base<T>& tensor, integers... ints) {
	auto index_point =  BC::make_array(ints...);
    return [&, index_point](auto... shape_indicies) {
        return make_tensor(et::make_chunk(
                tensor.internal_base(),
                index_point,
                BC::make_array(shape_indicies...)));
    };
}


}//end of BC name space


#endif /* TENSOR_SHAPING_H_ */
