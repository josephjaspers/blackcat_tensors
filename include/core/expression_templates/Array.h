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
namespace exprs {

template<class, int, bool> class Array_Slice;
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
		  Shape<Dimension>, public Tags... {

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

     value_type* array = nullptr;

    BCINLINE const value_type* memptr() const { return array; }
    BCINLINE       value_type* memptr()       { return array; }

    BCINLINE const Shape<Dimension>& get_shape() const { return static_cast<const Shape<Dimension>&>(*this); }

};

//specialization for scalar --------------------------------------------------------------------------------------------------------
template<class T, class Allocator, class... Tags>
struct ArrayExpression<0, T, Allocator, Tags...>
: Array_Base<ArrayExpression<0, T, Allocator, Tags...>, 0>,
  public Shape<0>,
  public Tags... {

	using derived_t = Array<0, T, Allocator, Tags...>;
	using value_type = T;
	using allocator_t = Allocator;
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;

	static constexpr int DIMS = 0;
	static constexpr int ITERATOR = 0;

	value_type* array = nullptr;

	BCINLINE const auto& operator [] (int index) const { return array[0]; }
	BCINLINE	   auto& operator [] (int index) 	   { return array[0]; }

	template<class... integers> BCINLINE
	const auto& operator () (integers... ints) const { return array[0]; }

	template<class... integers> BCINLINE
	      auto& operator () (integers... ints)      { return array[0]; }

	BCINLINE const value_type* memptr() const { return array; }
	BCINLINE       value_type* memptr()       { return array; }

    BCINLINE const Shape<0>& get_shape() const { return static_cast<const Shape<0>&>(*this); }


};


template<class,int,bool> class Array_Slice;

template<int Dimension, class Scalar, class Allocator, class... Tags>
class Array :
			private Allocator,
			public Context<typename BC::allocator_traits<Allocator>::system_tag>,
			public ArrayExpression<Dimension, Scalar, Allocator, Tags...> {

	template<class,int, bool> friend class Array_Slice;

	using self = Array<Dimension, Scalar, Allocator, Tags...>;
	using parent = ArrayExpression<Dimension, Scalar, Allocator, Tags...>;

public:

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using allocator_t = Allocator;
	using context_t   = Context<system_tag>;
	using full_context_t = context::full_context_t<allocator_t, context_t>;
	using value_type = Scalar;
	using ArrayExpression<Dimension, Scalar, Allocator, Tags...>::deallocate;

	const auto get_full_context() const {
		return context::make_full_context(as_alloc(), as_context());
	}

	auto get_full_context() {
		return context::make_full_context(as_alloc(), as_context());
	}

private:

	Shape<Dimension>& as_shape() { return static_cast<Shape<Dimension>&>(*this); }

	const Allocator& as_alloc() const { return static_cast<const Allocator&>(*this); }
		  Allocator& as_alloc() 	  { return static_cast<		 Allocator&>(*this); }

	const context_t& as_context() const { return static_cast<const context_t&>(*this); }
          context_t& as_context() 	    { return static_cast<	   context_t&>(*this); }

public:

	Array() {
		if (Dimension == 0) {
			this->array = this->allocate(1);
		}
	}

	Array(const Array& array_)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(array_)),
	  context_t(array_.as_context()),
	  parent(array_) {
        this->array = this->allocate(this->size());
        evaluate_to(this->internal(), array_.internal(), get_full_context());
	}
	Array(Array&& array_) //TODO handle propagate_on_container_move_assignment
	: Allocator(std::move(array_.as_alloc())),
	  context_t(std::move(array_.as_context())),
	  parent(array_) {
		array_.m_inner_shape = {0};
		array_.m_block_shape = {0};
		array_.array = nullptr;
	}

	//Construct via shape-like object
    template<class U,typename = std::enable_if_t<!expression_traits<U>::is_array && !expression_traits<U>::is_expr>>
    Array(U param) {
    	this->as_shape() = Shape<Dimension>(param);
    	this->array = this->allocate(this->size());
    }

	//Construct via shape-like object and Allocator
    template<class U,typename = std::enable_if_t<!expression_traits<U>::is_array && !expression_traits<U>::is_expr>>
    Array(U param, const Allocator& alloc_) : Allocator(alloc_) {
    	this->as_shape() = Shape<Dimension>(param);
    	this->array = this->allocate(this->size());
    }


	//Constructor for integer sequence, IE Matrix(m, n)
	template<class... args,
	typename=std::enable_if_t<
		meta::seq_of<BC::size_t, args...> &&
		sizeof...(args) == Dimension>>
	Array(const args&... params) {
		this->as_shape() = Shape<Dimension>(params...);
		this->array      = this->allocate(this->size());
	}

	//Shape-like object with maybe allocator
    template<class Expr,typename = std::enable_if_t<expression_traits<Expr>::is_array || expression_traits<Expr>::is_expr>>
	Array(const Expr& expr_t, const Allocator& alloc=Allocator()) : Allocator(alloc) {
		this->as_shape() = Shape<Dimension>(expr_t.inner_shape());
		this->array = this->allocate(this->size());
		evaluate_to(this->internal(), expr_t.internal(), get_full_context());
	}


	//If Copy-constructing from a slice, attempt to query the allocator
    //Restrict to same value_type (obviously), same dimensions (for fast-copy)
    //And restrict to continuous (as we should attempt to support Sparse matrices in the future)
    template<
    	class Parent,//>//,
    	typename=
    			std::enable_if_t<
    			std::is_same<typename Parent::allocator_t, allocator_t>::value &&
    			std::is_same<typename Parent::value_type, value_type>::value &&
    			Array_Slice<Parent, Dimension, true>::DIMS == Dimension>>
	Array(const Array_Slice<Parent, Dimension, true>& expr_t)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(
			expr_t.get_full_context().get_allocator())),
	  context_t(expr_t.get_full_context())
	{
		this->as_shape() = Shape<Dimension>(expr_t.inner_shape());
		this->array = this->allocate(this->size());
        evaluate_to(this->internal(), expr_t.internal(), this->get_full_context());
	}

public:
    void internal_swap(Array& swap) {
    	std::swap(this->array, swap.array);
    	std::swap(this->m_inner_shape, swap.m_inner_shape);
    	std::swap(this->m_block_shape, swap.m_block_shape);

    	if (BC::allocator_traits<Allocator>::propagate_on_container_swap::value) {
    		std::swap(this->as_alloc(), swap.as_alloc());
    	}
    }

    void deallocate() {
       Allocator::deallocate(this->array, this->size());
       this->array = nullptr;
    }

    	self& internal_base() {
    		return *this; }
    	const self& internal_base() const {
    		return *this; }


	Allocator get_allocator() const {
		return BC::allocator_traits<Allocator>::select_on_container_copy_construction(static_cast<const Allocator&>(*this));
	}
};


template<class Shape, class Allocator>
auto make_tensor_array(Shape shape, Allocator alloc) {
	return Array<Shape::DIMS, typename Allocator::value_type, Allocator>(shape, alloc);
}


}
}


#endif /* SHAPE_H_ */
