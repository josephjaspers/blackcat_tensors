/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_ARRAY_H_
#define BC_EXPRESSION_TEMPLATES_ARRAY_H_


#include "Expression_Template_Base.h"


namespace BC {
namespace tensors {
namespace exprs {

/*
 * 	Array is a class that inherits from Kernel_Array and the Allocator type.
 *  The Array class is used to initialize and destruct the Kernel_Array object.
 *
 *  Array and Kernel_Array are two tightly coupled classes as expression-templates must be trivially copyable (to pass them to CUDA functions).
 *  Separating these two enables the usage of non-trivially copyable allocators as well as the ability to define
 *  non-default move and copy assignments/constructors.
 *
 *  The Kernel_Array class should never be instantiated normally. It should only be accessed by instantiating
 *  an instance of the Array class, and calling 'my_array_object.internal()' to query it.
 *
 *  Additionally this design pattern (replicated in Array_View) enables expression-templates to
 *  defines additional members that we do not want to pass to the GPU. (As they my be non-essential to the computation).
 *
 */


template<int Dimension, class ValueType, class SystemTag, class... Tags>
struct Kernel_Array
		: Kernel_Array_Base<Kernel_Array<Dimension, ValueType, SystemTag, Tags...>>,
		  std::conditional_t<BC::traits::sequence_contains_v<BC_Noncontinuous, Tags...> && (Dimension>1), SubShape<Dimension>, Shape<Dimension>>,
		  public Tags... {

    using value_type = ValueType;
    using system_tag = SystemTag;
    using pointer_type = value_type*;
    using shape_type = std::conditional_t<BC::traits::sequence_contains_v<BC_Noncontinuous, Tags...> && (Dimension>1), SubShape<Dimension>, Shape<Dimension>>;
    using self_type  = Kernel_Array<Dimension, ValueType, SystemTag, Tags...>;

	static constexpr bool self_is_view = BC::traits::sequence_contains_v<BC_View, Tags...>;
	static constexpr bool is_continuous = ! BC::traits::sequence_contains_v<BC_Noncontinuous, Tags...>;

	static constexpr bool copy_constructible = !self_is_view;
	static constexpr bool move_constructible = !self_is_view;
    static constexpr bool copy_assignable    = true;
	static constexpr bool move_assignable    = !self_is_view;

    static constexpr int tensor_dimension = Dimension;
    static constexpr int tensor_iterator_dimension = is_continuous ? 1 : tensor_dimension;

private:
    pointer_type array = nullptr;

protected:

    BCINLINE pointer_type& memptr_ref() { return array; }
    BCINLINE shape_type& get_shape_ref() { return static_cast<shape_type&>(*this); }

public:
    Kernel_Array()=default;
    Kernel_Array(const Kernel_Array&)=default;
    Kernel_Array(Kernel_Array&&)=default;
    Kernel_Array(shape_type shape, pointer_type ptr)
    	: shape_type(shape), array(ptr) {};

    template<class AllocatorType>
    Kernel_Array(shape_type shape, AllocatorType& allocator)
    	: shape_type(shape), array(allocator.allocate(this->size())) {};

    BCINLINE pointer_type memptr() const { return array; }
    BCINLINE const shape_type& get_shape() const { return static_cast<const shape_type&>(*this); }

    BCINLINE const auto& operator [](int index) const {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else if (!expression_traits<self_type>::is_continuous && tensor_dimension==1) {
    		return array[this->leading_dimension(0) * index];
    	} else {
    		return array[index];
    	}
    }

    BCINLINE auto& operator [](int index) {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else if (!expression_traits<self_type>::is_continuous && tensor_dimension==1) {
    		return array[this->leading_dimension(0) * index];
    	} else {
    		return array[index];
    	}
    }

    template<class ... integers>
    BCINLINE const auto& operator ()(integers ... ints) const {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else {
    		return array[this->dims_to_index(ints...)];
    	}
    }

    template<class ... integers>
    BCINLINE auto& operator ()(integers ... ints) {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else {
    		return array[this->dims_to_index(ints...)];
    	}
    }

    BCINLINE auto slice_ptr_index(int i) const {
        if (tensor_dimension == 0)
            return 0;
        else if (tensor_dimension == 1)
            return i;
        else
            return this->leading_dimension(Dimension - 2) * i;
    }

};


template<int, class, class, class...>
class Array_Slice;


template<int Dimension, class Scalar, class Allocator, class... Tags>
class Array :
			private Allocator,
			public Kernel_Array<Dimension, Scalar, typename BC::allocator_traits<Allocator>::system_tag, Tags...> {

	using self = Array<Dimension, Scalar, Allocator, Tags...>;
	using parent = Kernel_Array<Dimension, Scalar,  typename BC::allocator_traits<Allocator>::system_tag, Tags...>;
	using stream_type = Stream<typename BC::allocator_traits<Allocator>::system_tag>;

public:

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using allocator_t = Allocator;
	using stream_t   = Stream<system_tag>;
	using value_type = Scalar;

private:

	stream_type m_stream;


	const Shape<Dimension>& as_shape() const { return static_cast<const Shape<Dimension>&>(*this); }
	Shape<Dimension>& as_shape() { return static_cast<Shape<Dimension>&>(*this); }
	Allocator& get_allocator_ref() { return static_cast<Allocator&>(*this); }


public:
	const stream_t& get_stream()   const { return m_stream; }
	stream_t& get_stream()   { return m_stream; }

	Allocator get_allocator() const { return static_cast<const Allocator&>(*this); }

	Array() {
		if (Dimension == 0) {
			this->memptr_ref() = get_allocator().allocate(1);
		}
	}

	Array(const Array& array)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(array)),
	  parent(array.get_shape(), get_allocator_ref()),
	  m_stream(array.get_stream()) {
		greedy_evaluate(this->internal(), array.internal(), get_stream());
	}

	Array(Array&& array) //TODO handle propagate_on_container_move_assignment
	: Allocator(array.get_allocator()),
	  parent(array),
	  m_stream(array.get_stream()) {
		array.memptr_ref() = nullptr;
		//This causes segmentation fault with NVCC currently (compiler segfault, not runtime)
//		array_.as_shape() = BC::Shape<Dimension>(); //resets the shape
	}

	//Construct via shape-like object and Allocator
    template<
    	class ShapeLike,
    	class=std::enable_if_t<
    		!expression_traits<ShapeLike>::is_array &&
    		!expression_traits<ShapeLike>::is_expr &&
    		Dimension != 0>
    >
    Array(ShapeLike param, Allocator allocator=Allocator()):
    	Allocator(allocator),
    	parent(typename parent::shape_type(param), get_allocator_ref()) {}

	//Constructor for integer sequence, IE Matrix(m, n)
	template<
		class... ShapeDims,
		class=std::enable_if_t<
			traits::sequence_of_v<BC::size_t, ShapeDims...> &&
			sizeof...(ShapeDims) == Dimension>
	>
	Array(const ShapeDims&... shape_dims):
		parent(typename parent::shape_type(shape_dims...), get_allocator_ref()) {}

	//Shape-like object with maybe allocator
    template<
    	class Expression,
    	class=std::enable_if_t<expression_traits<Expression>::is_array || expression_traits<Expression>::is_expr>>
	Array(const Expression& expression, Allocator allocator=Allocator()):
		Allocator(allocator),
		parent(typename parent::shape_type(expression.inner_shape()), get_allocator_ref()) {
		greedy_evaluate(this->internal(), expression.internal(), get_stream());
	}


	//If Copy-constructing from a slice, attempt to query the allocator
    //Restrict to same value_type (obviously), same dimensions (for fast-copy)
    //And restrict to continuous (as we should attempt to support Sparse matrices in the future)
    template<class... SliceTags>
	Array(const Array_Slice<Dimension, value_type, allocator_t, SliceTags...>& expression):
		Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(expression.get_allocator())),
		parent(typename parent::shape_type(expression.inner_shape()), get_allocator_ref()),
		m_stream(expression.get_stream()) {
		greedy_evaluate(this->internal(), expression.internal(), this->get_stream());
	}

public:
    Array& operator = (Array&& array) {
    		if (BC::allocator_traits<Allocator>::is_always_equal::value ||
    			array.get_allocator() == this->get_allocator()) {
			std::swap(this->memptr_ref(), array.memptr_ref());
			this->as_shape() = array.as_shape();

	    	if (BC::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
				static_cast<Allocator&>(*this) = static_cast<Allocator&&>(array);
			}
    	} else {
    		get_allocator().deallocate(this->memptr_ref(), this->size());
			this->as_shape() = array.as_shape();
			this->memptr_ref() = get_allocator().allocate(this->size());
			greedy_evaluate(this->internal(), array.internal(), this->get_stream());
    	}
    		return *this;
    }

    void deallocate() {
       Allocator::deallocate(this->memptr_ref(), this->size());
       this->memptr_ref() = nullptr;
	}

};

template<class ValueType, int dims, class Stream>
auto make_temporary_kernel_array(Shape<dims> shape, Stream stream) {
//	static_assert(dims >= 1, "make_temporary_tensor_array: assumes dimension is 1 or greater");
	using system_tag = typename Stream::system_tag;
	using Array = Kernel_Array<dims, ValueType, system_tag, BC_Temporary>;
	return Array(shape, stream.template get_allocator_rebound<ValueType>().allocate(shape.size()));
}
template<class ValueType, class Stream>
auto make_temporary_kernel_scalar(Stream stream) {
	using system_tag = typename Stream::system_tag;
	using Array = Kernel_Array<0, ValueType, system_tag, BC_Temporary>;
	return Array(BC::Shape<0>(), stream.template get_allocator_rebound<ValueType>().allocate(1));
}

template<
	int Dimension,
	class ValueType,
	class Stream,
	class... Tags,
	class=std::enable_if_t<BC::traits::sequence_contains_v<BC_Temporary, Tags...>>>
void destroy_temporary_kernel_array(
		Kernel_Array<Dimension, ValueType, typename Stream::system_tag, Tags...> temporary, Stream stream) {
	stream.template get_allocator_rebound<ValueType>().deallocate(temporary.memptr(), temporary.size());
}


template<class Shape, class Allocator>
auto make_tensor_array(Shape shape, Allocator alloc) {
	return Array<Shape::tensor_dimension, typename Allocator::value_type, Allocator>(shape, alloc);
}


} //ns BC
} //ns exprs
} //ns tensors



#endif /* SHAPE_H_ */
