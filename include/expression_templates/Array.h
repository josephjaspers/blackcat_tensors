/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_H_

#include "Array_Base.h"

namespace BC {
namespace et {


template<int Dimension, class Scalar, class Allocator>
struct Array
		: Array_Base<Array<Dimension, Scalar, Allocator>, Dimension>,
		  Shape<Dimension>,
		  private Allocator {

	static_assert(std::is_trivially_copyable<Allocator>::value,
			"BC_TENSOR_ALLOCATOR MUST BE TRIVIALLY COPYABLE");

    using value_type = Scalar;
    using allocator_t = Allocator;
    using system_tag = typename allocator_t::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = true;

    static constexpr int  DIMS = Dimension;
    static constexpr int ITERATOR = 1;

    value_type* array = nullptr;
    Array() = default;

    Array(Shape<DIMS> shape_, value_type* array_) : array(array_), Shape<DIMS>(shape_) {}

    template<class U,typename = std::enable_if_t<not std::is_base_of<BC_internal_interface<U>, U>::value>>
    Array(U param) : Shape<DIMS>(param), array( allocator_t::allocate(this->size())) {}

    template<class... integers>//CAUSES FAILURE WITH NVCC 9.2, typename = std::enable_if_t<MTF::is_integer_sequence<integers...>>>
    Array(integers... ints) : Shape<DIMS>(ints...), array(allocator_t::allocate(this->size())) {
        static_assert(MTF::seq_of<int, integers...>,"PARAMETER LIST MUST BE INTEGER_SEQUNCE");
    }

    template<class deriv_expr, typename = std::enable_if_t<std::is_base_of<BC_internal_interface<deriv_expr>, deriv_expr>::value>>
    Array(const deriv_expr& expr) : Shape<DIMS>(static_cast<const deriv_expr&>(expr).inner_shape()),
    array(allocator_t::allocate(this->size())){
        evaluate_to(*this, expr);
    }

protected:
    template<class U> Array(U param, value_type* array_) : array(array_), Shape<DIMS>(param) {}
    Array(value_type* array_) : array(array_) {}


    void copy_init(const Array& array_copy) {
        this->copy_shape(array_copy);
        this->array = allocator_t::allocate(this->size());
        evaluate_to(*this, array_copy);
    }
public:
    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }

    void swap_array(Array& param) {
        std::swap(this->array, param.array);
    }

    void deallocate() {
        allocator_t::deallocate(array, this->size());
        array = nullptr;
    }
};


//specialization for scalar --------------------------------------------------------------------------------------------------------
template<class T, class allocator>
struct Array<0, T, allocator> : Array_Base<Array<0, T, allocator>, 0>, public Shape<0>, private allocator {

    using value_type = T;
    using allocator_t = allocator;
    using system_tag = typename allocator_t::system_tag;

    static constexpr int DIMS = 0;
    static constexpr int ITERATOR = 0;


    value_type* array = nullptr;


    Array()
     : array(allocator_t::allocate(this->size())) {}

    Array(Shape<DIMS> shape_, value_type* array_)
    : array(array_), Shape<0>(shape_) {}

    template<class U>
    Array(U param) {
    	allocator_t::allocate(array, this->size());
    	evaluate_to(*this, param);
    }

    template<class U>
    Array(U param, value_type* array_) : array(array_), Shape<DIMS>(param) {}

    __BCinline__
    const auto& operator [] (int index) const {
    	return array[0];
    }

    __BCinline__
    auto& operator [] (int index) {
    	return array[0];
    }

    template<class... integers> __BCinline__
    auto& operator () (integers... ints) {
        return array[0];
    }

    template<class... integers> __BCinline__
    const auto& operator () (integers... ints) const {
        return array[0];
    }

    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }

    void copy_init(const Array& array_copy) {
        allocator_t::allocate(array, this->size());
        evaluate_to(*this, array_copy);
    }

    void swap_array(Array& param) {
        std::swap(this->array, param.array);
    }

    void deallocate() {
        allocator_t::deallocate(array, this->size());
        array = nullptr;
    }

};


}
}


#endif /* SHAPE_H_ */
