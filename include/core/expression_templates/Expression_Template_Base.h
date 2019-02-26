/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_INTERNAL_BASE_H_
#define BC_INTERNAL_BASE_H_

#include "Common.h"

namespace BC {
namespace expression_templates {

template<class derived>
class Expression_Template_Base : BC_Type {

    static constexpr int  DIMS = derived::DIMS;

    BCINLINE const derived& as_derived() const { return static_cast<const derived&>(*this); }
    BCINLINE       derived& as_derived()       { return static_cast<      derived&>(*this); }

public:

    BCINLINE const auto& internal() const { return as_derived(); }
    BCINLINE       auto& internal()       { return as_derived(); }

    //default method - returns the mixin of Tensor_Base (may be different than internal())
    //should be shadowed by classes that support non-trivial copy/move construction
    BCINLINE const auto& internal_base() const { return as_derived(); }
    BCINLINE       auto& internal_base()       { return as_derived(); }


    static constexpr bool copy_constructible = false;
    static constexpr bool move_constructible = false;
    static constexpr bool copy_assignable    = false;
    static constexpr bool move_assignable    = false;

    operator       derived&()       { return as_derived(); }
    operator const derived&() const { return as_derived(); }

    BCINLINE Expression_Template_Base() {
	static_assert(std::is_trivially_copy_constructible<derived>::value, "INTERNAL_TYPES TYPES MUST BE TRIVIALLY COPYABLE");
	static_assert(std::is_trivially_copyable<derived>::value, "INTERNAL_TYPES MUST BE TRIVIALLY COPYABLE");
	static_assert(!std::is_same<void, typename derived::value_type>::value, "INTERNAL_TYPES MUST HAVE A 'using value_type = some_Type'");
	static_assert(!std::is_same<decltype(std::declval<derived>().inner_shape()), void>::value, "INTERNAL_TYPE MUST DEFINE inner_shape()");
	static_assert(!std::is_same<decltype(std::declval<derived>().block_shape()), void>::value, "INTERNAL_TYPE MUST DEFINE block_shape()");
	static_assert(std::is_same<decltype(std::declval<derived>().rows()), int>::value, "INTERNAL_TYPE MUST DEFINE rows()");
	static_assert(std::is_same<decltype(std::declval<derived>().cols()), int>::value, "INTERNAL_TYPE MUST DEFINE cols()");
	static_assert(std::is_same<decltype(std::declval<derived>().dimension(0)), int>::value, "INTERNAL_TYPE MUST DEFINE dimension(int)");
	static_assert(std::is_same<decltype(std::declval<derived>().block_dimension(0)), int>::value, "INTERNAL_TYPE MUST DEFINE block_dimension(int)");
    }

    void deallocate() const {}

};


}
}


#endif /* BC_INTERNAL_BASE_H_ */
