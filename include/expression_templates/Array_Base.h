/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "Internal_Type_Interface.h"
#include "Internal_Common.h"
#include "Shape.h"

namespace BC {
namespace et     {

template<class derived, int DIMENSION>
struct Array_Base : BC_internal_interface<derived>, BC_Array {

    __BCinline__ static constexpr int DIMS() { return DIMENSION; }
    __BCinline__ static constexpr int ITERATOR() { return 0; }

    using self = derived;

private:

    __BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
    __BCinline__       derived& as_derived()       { return static_cast<      derived&>(*this); }

public:

    static constexpr bool copy_constructible = false;
    static constexpr bool move_constructible = false;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = false;

    __BCinline__ operator const auto*() const { return as_derived().memptr(); }
    __BCinline__ operator       auto*()       { return as_derived().memptr(); }

    __BCinline__
    const auto& operator [](int index) const {
        return as_derived().memptr()[index];
    }
    __BCinline__
    auto& operator [](int index) {
        return as_derived().memptr()[index];
    }

    template<class ... integers>
    __BCinline__ const auto& operator ()(integers ... ints) const {
        return as_derived()[this->dims_to_index(ints...)];
    }
    template<class ... integers>
    __BCinline__ auto& operator ()(integers ... ints) {
        return as_derived()[this->dims_to_index(ints...)];
    }


    template<int length>
    __BCinline__ auto& operator ()(const BC::array<length, int>& index) {
        static_assert(length >= DIMS(), "POINT MUST HAVE AT LEAST THE SAME NUMBER OF INDICIES AS THE TENSOR");
        return as_derived()[this->dims_to_index(index)];
    }

    template<int length>
    __BCinline__ const auto& operator ()(const BC::array<length, int>& index) const {
        static_assert(length >= DIMS(), "POINT MUST HAVE AT LEAST THE SAME NUMBER OF INDICIES AS THE TENSOR");
        return as_derived()[this->dims_to_index(index)];
    }

    void deallocate() {}
    //------------------------------------------Implementation Details---------------------------------------//
public:

    __BCinline__
    auto slice_ptr(int i) const {
        if (DIMS() == 0)
            return &as_derived()[0];
        else if (DIMS() == 1)
            return &as_derived()[i];
        else
            return &as_derived()[as_derived().leading_dimension(DIMENSION - 2) * i];
    }

private:

    template<class... integers> __BCinline__
    int dims_to_index(integers... ints) const {
        return dims_to_index(BC::make_array(ints...));
    }

    template<int D> __BCinline__
    int dims_to_index(const BC::array<D, int>& var) const {
        int index = var[0];
        for(int i = 1; i < DIMS(); ++i) {
            index += this->as_derived().leading_dimension(i - 1) * var[i];
        }
        return index;
    }
};
//------------------------------------------------type traits--------------------------------------------------------------//

template<class T> static constexpr bool is_array() { return std::is_base_of<et::Array_Base<T, T::DIMS()>, T>::value; };
template<class T> static constexpr bool is_expr()  { return !is_array<T>(); };

}
}


#endif /* TENSOR_CORE_INTERFACE_H_ */

