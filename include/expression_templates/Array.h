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


template<int,class,class,class...> class Array; //derived


/*
 * 	Array is a class that inherits from ArrayExpression and the Allocator type.
 *  The Array class is used to initialize and destruct the ArrayExpression object.
 *
 *  Array and ArrayExpression are two tightly coupled classes as expression-templates must be trivially copyable (to pass them to CUDA functions).
 *  Separating these two enables the usage of non-trivially copyable allocators as well as the ability to define
 *  non-default move and copy assignments/constructors.
 *
 *  The ArrayExpression class should never be instantiated normally. It should only be accessed by instantiating
 *  an instance of the Array class, and calling 'my_array_object.internal()' to query it.
 *
 *  Additionally this design pattern (replicated in Array_View) enables expression-templates to
 *  defines additional members that we do not want to pass to the GPU. (As they my be non-essential to the computation).
 *
 */


template<int Dimension, class Scalar, class Allocator, class... Tags>
struct ArrayExpression
		: Array_Base<ArrayExpression<Dimension, Scalar, Allocator, Tags...>, Dimension>,
		  Shape<Dimension> {

	using derived_t = Array<Dimension, Scalar, Allocator, Tags...>;
    using value_type = Scalar;
    using allocator_t = Allocator;
    using system_tag = typename allocator_traits<Allocator>::system_tag;

    static constexpr bool copy_constructible = true;
    static constexpr bool move_constructible = true;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = true;

    static constexpr int  DIMS = Dimension;
    static constexpr int ITERATOR = 1;

private:
    const auto& as_derived() const { return static_cast<const derived_t&>(*this); }
          auto& as_derived()  	   { return static_cast<	  derived_t&>(*this); }
public:
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

    void swap_init(ArrayExpression& array_move) {
    	std::swap(this->array, array_move.array);
    	this->swap_shape(array_move);
    }

public:
    __BCinline__ const value_type* memptr() const { return array; }
    __BCinline__       value_type* memptr()       { return array; }


    void deallocate() {
        get_allocator().deallocate(array, this->size());
        array = nullptr;
    }
};



template<int Dimension, class Scalar, class Allocator, class... Tags>
class Array :
			private Allocator,
			public ArrayExpression<Dimension, Scalar, Allocator, Tags...> {

	template<int, class, class, class...>
	friend class ArrayExpression;

	using self = Array<Dimension, Scalar, Allocator, Tags...>;
	using parent = ArrayExpression<Dimension, Scalar, Allocator, Tags...>;

public:

	using allocator_t = Allocator;
	using internal_t = ArrayExpression<Dimension, Scalar, Allocator, Tags...>;
	using value_type = Scalar;
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;

	using ArrayExpression<Dimension, Scalar, Allocator, Tags...>::deallocate;

	Array() = default;


	Array(const Allocator& alloc)
	: allocator_t(BC::allocator_traits<Allocator>::select_on_container_copy_construction(alloc)) {
	}

	template<class... args, typename=std::enable_if_t<MTF::seq_of<BC::size_t, args...>>>
	Array(const args&... params)
	: allocator_t(allocator_t()),
	  parent(make_array(params...)){}

	Array(const parent& parent_)
	: parent(parent_) {}

	Array(parent&& parent_)
	: parent(parent_) {}
};


//specialization for scalar --------------------------------------------------------------------------------------------------------
template<class T, class Allocator, class... Tags>
struct ArrayExpression<0, T, Allocator, Tags...>
: Array_Base<ArrayExpression<0, T, Allocator, Tags...>, 0>, public Shape<0> {

	using derived_t = Array<0, T, Allocator, Tags...>;
    using value_type = T;
    using allocator_t = Allocator;
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;

    static constexpr int DIMS = 0;
    static constexpr int ITERATOR = 0;

    value_type* array = nullptr;

private:
    const auto& as_derived() const { return static_cast<const derived_t&>(*this); }
          auto& as_derived()  	   { return static_cast<	  derived_t&>(*this); }
public:
    const auto& get_allocator() const { return static_cast<const Allocator&>(as_derived()); }
          auto& get_allocator() 	  { return static_cast<	     Allocator&>(as_derived()); }

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

    void swap_init(const ArrayExpression& array_move) {
        	std::swap(this->array, array_move.array);
        	this->swap_shape(array_move);
	}

    void deallocate() {
        get_allocator().deallocate(this->array, this->size());
        array = nullptr;
    }

};


}
}


#endif /* SHAPE_H_ */
