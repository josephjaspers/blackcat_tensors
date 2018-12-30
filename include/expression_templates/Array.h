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

template<int, class, class> class Array;

template<int Dimension, class Scalar, class Allocator, class AllocatorBase>
struct ArrayExpression
		: Array_Base<ArrayExpression<Dimension, Scalar, Allocator, AllocatorBase>, Dimension>,
		  Shape<Dimension> {

    using value_type = Scalar;
    using allocator_t = Allocator;
    using system_tag = typename Allocator::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = true;

    static constexpr int  DIMS = Dimension;
    static constexpr int ITERATOR = 1;

private:
    const auto& as_derived() const { return static_cast<const AllocatorBase&>(*this); }
          auto& as_derived()  	   { return static_cast<	  AllocatorBase&>(*this); }

    const auto& get_allocator() const { return static_cast<const Allocator&>(as_derived()); }
          auto& get_allocator() 	  { return static_cast<	     Allocator&>(as_derived()); }
public:
    value_type* array = nullptr;
    ArrayExpression() = default;

    ArrayExpression(Shape<DIMS> shape_, value_type* array_) : array(array_), Shape<DIMS>(shape_) {}

    template<class U,typename = std::enable_if_t<! std::is_base_of<BC_internal_interface<U>, U>::value>>
	ArrayExpression(U param) : Shape<DIMS>(param), array(get_allocator().allocate(this->size())) {}

    template<class... integers>//CAUSES FAILURE WITH NVCC 9.2, typename = std::enable_if_t<MTF::is_integer_sequence<integers...>>>
    ArrayExpression(integers... ints) : Shape<DIMS>(ints...), array(get_allocator().allocate(this->size())) {
        static_assert(MTF::seq_of<int, integers...>,"PARAMETER LIST MUST BE INTEGER_SEQUNCE");
    }

    template<class deriv_expr, typename = std::enable_if_t<std::is_base_of<BC_internal_interface<deriv_expr>, deriv_expr>::value>>
    ArrayExpression(const deriv_expr& expr) : Shape<DIMS>(static_cast<const deriv_expr&>(expr).inner_shape()),
    array(get_allocator().allocate(this->size())){
        evaluate_to(*this, expr);
    }

protected:
    template<class U>
    ArrayExpression(U param, value_type* array_) : array(array_), Shape<DIMS>(param) {}
    ArrayExpression(value_type* array_) : array(array_) {}


    void copy_init(const ArrayExpression& array_copy) {
        this->copy_shape(array_copy);
        this->array = get_allocator().allocate(this->size());
        evaluate_to(*this, array_copy);
    }
public:
    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }

    void swap_array(ArrayExpression& param) {
        std::swap(this->array, param.array);
    }

    void deallocate() {
        get_allocator().deallocate(array, this->size());
        array = nullptr;
    }
};

template<int Dimension, class Scalar, class Allocator>
class Array :
		private std::conditional_t<
			allocator::has_system_tag<Allocator>::value,
			Allocator,
			allocator::CustomAllocator<Allocator>>,

			public ArrayExpression<Dimension, Scalar, std::conditional_t<
														allocator::has_system_tag<Allocator>::value,
														Allocator,
														allocator::CustomAllocator<Allocator>>, Array<Dimension, Scalar, Allocator>>
	{

	template<int, class, class, class>
	friend class ArrayExpression;


public:

	using self = Array<Dimension, Scalar, Allocator>;
	using allocator_t = std::conditional_t<allocator::has_system_tag<Allocator>::value, Allocator, allocator::CustomAllocator<Allocator>>;
	using internal_t = ArrayExpression<Dimension, Scalar, allocator_t, self>;
	using value_type = Scalar;
	using system_tag = typename allocator_t::system_tag;

	using ArrayExpression<Dimension, Scalar, allocator_t, self>::deallocate;

	Array() = default;

	template<class... args>
	Array(Allocator alloc, const args&... params)
	: allocator_t(alloc), internal_t(params...) {}

	template<class... args>
	Array(const args&... params)
	: allocator_t(allocator_t()), internal_t(params...) {}
};



template<int Dimension, class Scalar, class Allocator>
class Temporary :
		private std::conditional_t<
			allocator::has_system_tag<Allocator>::value,
			Allocator,
			allocator::CustomAllocator<Allocator>>,

			public ArrayExpression<Dimension, Scalar, std::conditional_t<
														allocator::has_system_tag<Allocator>::value,
														Allocator,
														allocator::CustomAllocator<Allocator>>, Temporary<Dimension, Scalar, Allocator>>
	{

	template<int, class, class, class>
	friend class ArrayExpression;


public:

	using self = Temporary<Dimension, Scalar, Allocator>;
	using allocator_t = std::conditional_t<allocator::has_system_tag<Allocator>::value, Allocator, allocator::CustomAllocator<Allocator>>;
	using internal_t = ArrayExpression<Dimension, Scalar, allocator_t, self>;
	using value_type = Scalar;
	using system_tag = typename allocator_t::system_tag;

	using ArrayExpression<Dimension, Scalar, allocator_t, self>::deallocate;

	Temporary() = default;

	template<class... args>
	Temporary(Allocator alloc, const args&... params)
	: allocator_t(alloc), internal_t(params...) {}

	template<class... args>
	Temporary(const args&... params)
	: allocator_t(allocator_t()), internal_t(params...) {}
};

//specialization for scalar --------------------------------------------------------------------------------------------------------
template<class T, class Allocator, class AllocatorBase>
struct ArrayExpression<0, T, Allocator, AllocatorBase> : Array_Base<ArrayExpression<0, T, Allocator, AllocatorBase>, 0>, public Shape<0> {

    using value_type = T;
    using allocator_t = Allocator;
    using system_tag = typename Allocator::system_tag;

    static constexpr int DIMS = 0;
    static constexpr int ITERATOR = 0;

    value_type* array = nullptr;

private:
    const auto& as_derived() const { return static_cast<const AllocatorBase&>(*this); }
          auto& as_derived()  	   { return static_cast<	  AllocatorBase&>(*this); }

    const auto& get_allocator() const { return static_cast<const Allocator&>(as_derived()); }
          auto& get_allocator() 	  { return static_cast<	     Allocator&>(as_derived()); }
public:

    ArrayExpression()
     : array(get_allocator().allocate(this->size())) {}

    ArrayExpression(Shape<DIMS> shape_, value_type* array_)
    : array(array_), Shape<0>(shape_) {}

    template<class U>
    ArrayExpression(U param) {
    	array = get_allocator().allocate(this->size());
    	evaluate_to(*this, param);
    }

    template<class U>
    ArrayExpression(U param, value_type* array_) : array(array_), Shape<DIMS>(param) {}

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

    void copy_init(const ArrayExpression& array_copy) {
        array = get_allocator().allocate(this->size());
        evaluate_to(*this, array_copy);
    }

    void swap_array(ArrayExpression& param) {
        std::swap(this->array, param.array);
    }

    void deallocate() {
        get_allocator().deallocate(this->array, this->size());
        array = nullptr;
    }

};


}
}


#endif /* SHAPE_H_ */
