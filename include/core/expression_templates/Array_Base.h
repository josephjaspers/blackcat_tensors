/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_EXPRESSION_TEMPLATES_ARRAY_BASE_H_
#define BLACKCAT_EXPRESSION_TEMPLATES_ARRAY_BASE_H_

#include "Internal_Type_Interface.h"
#include "Common.h"
#include "Shape.h"

namespace BC {
namespace et     {

template<class Derived, BC::dim_t Dimension>
struct Array_Base : BC_internal_interface<Derived>, BC_Array {

    static constexpr bool copy_constructible = false;
    static constexpr bool move_constructible = false;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = false;

    static constexpr int DIMS 		= Dimension;
    static constexpr int ITERATOR   = 0;

    using self = Derived;

private:

    BCINLINE const Derived& as_derived() const { return static_cast<const Derived&>(*this); }
    BCINLINE       Derived& as_derived()       { return static_cast<      Derived&>(*this); }

public:

    BCINLINE operator const auto*() const { return as_derived().memptr(); }
    BCINLINE operator       auto*()       { return as_derived().memptr(); }

    BCINLINE const auto& operator [](int index) const {
        return as_derived().memptr()[index];
    }

    BCINLINE auto& operator [](int index) {
        return as_derived().memptr()[index];
    }

    template<class ... integers>
    BCINLINE const auto& operator ()(integers ... ints) const {
        return as_derived()[this->dims_to_index(ints...)];
    }

    template<class ... integers>
    BCINLINE auto& operator ()(integers ... ints) {
        return as_derived()[this->dims_to_index(ints...)];
    }


    template<int length> BCINLINE
    auto& operator ()(const BC::array<length, int>& index) {
        static_assert(length >= DIMS,
        		"POINT MUST HAVE AT LEAST THE SAME NUMBER OF INDICIES AS THE TENSOR");

        return as_derived()[this->dims_to_index(index)];
    }

    template<int length> BCINLINE
    auto& operator ()(const BC::array<length, int>& index) const {
        static_assert(length >= DIMS,
        		"POINT MUST HAVE AT LEAST THE SAME NUMBER OF INDICIES AS THE TENSOR");

        return as_derived()[this->dims_to_index(index)];
    }

    void deallocate() {}

    //------------------------------------------Implementation Details---------------------------------------//

public:

    BCINLINE
    auto slice_ptr(int i) const {
        if (DIMS == 0)
            return &as_derived()[0];
        else if (DIMS == 1)
            return &as_derived()[i];
        else
            return &as_derived()[as_derived().leading_dimension(Dimension - 2) * i];
    }

    BCINLINE
    auto slice_ptr_index(int i) const {
        if (DIMS == 0)
            return 0;
        else if (DIMS == 1)
            return i;
        else
            return as_derived().leading_dimension(Dimension - 2) * i;
    }

    template<class... integers, typename=std::enable_if_t<meta::seq_of<BC::size_t, integers...>>> BCINLINE
    BC::size_t dims_to_index(integers... ints) const {
        return dims_to_index(BC::make_array(ints...));
    }

    template<int D> BCINLINE
    BC::size_t dims_to_index(const BC::array<D, int>& var) const {
        BC::size_t  index = var[0];
        for(int i = 1; i < DIMS; ++i) {
            index += this->as_derived().leading_dimension(i - 1) * var[i];
        }
        return index;
    }

};


}
}


#endif /* TENSOR_CORE_INTERFACE_H_ */

