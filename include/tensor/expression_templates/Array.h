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
		  std::conditional_t<BC::meta::sequence_contains_v<BC_Noncontinuous, Tags...> && (Dimension>1), SubShape<Dimension>, Shape<Dimension>>,
		  public Tags... {

    using value_type = ValueType;
    using system_tag = SystemTag;
    using pointer_type = value_type*;
    using shape_type = std::conditional_t<BC::meta::sequence_contains_v<BC_Noncontinuous, Tags...> && (Dimension>1), SubShape<Dimension>, Shape<Dimension>>;
    using self_type  = Kernel_Array<Dimension, ValueType, SystemTag, Tags...>;

    static constexpr bool copy_constructible = !expression_traits<self_type>::is_view;
    static constexpr bool move_constructible = !expression_traits<self_type>::is_view;
    static constexpr bool copy_assignable    = true;
    static constexpr bool move_assignable    = !expression_traits<self_type>::is_view;

    static constexpr int tensor_dimension = Dimension;
    static constexpr int tensor_iterator_dimension = expression_traits<self_type>::is_continuous ? 1 : tensor_dimension;

    pointer_type array = nullptr;

    Kernel_Array()=default;
    Kernel_Array(const Kernel_Array&)=default;
    Kernel_Array(Kernel_Array&&)=default;
    Kernel_Array(shape_type shape, pointer_type ptr)
    	: shape_type(shape), array(ptr) {};

    BCINLINE BC::meta::apply_const_t<pointer_type> memptr() const { return array; }
    BCINLINE       pointer_type memptr()       { return array; }

    BCINLINE const shape_type& get_shape() const { return static_cast<const shape_type&>(*this); }

    BCINLINE const auto& operator [](int index) const {
    	if (!expression_traits<self_type>::is_continuous && tensor_dimension==1) {
    		return array[this->leading_dimension(0) * index];
    	} else {
    		return array[index];
    	}
    }

    BCINLINE auto& operator [](int index) {
    	if (!expression_traits<self_type>::is_continuous && tensor_dimension==1) {
    		return array[this->leading_dimension(0) * index];
    	} else {
    		return array[index];
    	}
    }

    template<class ... integers>
    BCINLINE const auto& operator ()(integers ... ints) const {
        return array[this->dims_to_index(ints...)];
    }

    template<class ... integers>
    BCINLINE auto& operator ()(integers ... ints) {
        return array[this->dims_to_index(ints...)];
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


template<int ,class,class, class...> class Array_Slice;

template<int Dimension, class Scalar, class Allocator, class... Tags>
class Array :
			private Allocator,
			public Stream<typename BC::allocator_traits<Allocator>::system_tag>,
			public Kernel_Array<Dimension, Scalar, typename BC::allocator_traits<Allocator>::system_tag, Tags...> {

	using self = Array<Dimension, Scalar, Allocator, Tags...>;
	using parent = Kernel_Array<Dimension, Scalar,  typename BC::allocator_traits<Allocator>::system_tag, Tags...>;

public:

	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using allocator_t = Allocator;
	using stream_t   = Stream<system_tag>;
	using value_type = Scalar;

private:

	const Shape<Dimension>& as_shape() const { return static_cast<const Shape<Dimension>&>(*this); }
	Shape<Dimension>& as_shape() { return static_cast<Shape<Dimension>&>(*this); }


public:
	const stream_t& get_stream()   const { return static_cast<const stream_t&>(*this); }
	stream_t& get_stream()   { return static_cast<stream_t&>(*this); }

	Allocator get_allocator() const { return static_cast<const Allocator&>(*this); }

	Array() {
		if (Dimension == 0) {
			this->array = this->allocate(1);
		}
	}

	Array(const Array& array_)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(array_)),
	  stream_t(array_.get_stream()),
	  parent(array_) {
        this->array = this->allocate(this->size());
        greedy_evaluate(this->internal(), array_.internal(), get_stream());
	}
	Array(Array&& array_) //TODO handle propagate_on_container_move_assignment
	: Allocator(array_.get_allocator()),
	  stream_t(array_.get_stream()),
	  parent(array_) {
		array_.array = nullptr;
		//This causes segmentation fault with NVCC currently (compiler segfault, not runtime)
//		array_.as_shape() = BC::Shape<Dimension>(); //resets the shape
	}

	//Construct via shape-like object
    template<class U,typename = std::enable_if_t<!expression_traits<U>::is_array &&
    												!expression_traits<U>::is_expr &&
    													Dimension != 0>>
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
		meta::sequence_of_v<BC::size_t, args...> &&
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
		greedy_evaluate(this->internal(), expr_t.internal(), get_stream());
	}



	//If Copy-constructing from a slice, attempt to query the allocator
    //Restrict to same value_type (obviously), same dimensions (for fast-copy)
    //And restrict to continuous (as we should attempt to support Sparse matrices in the future)
    template<class... SliceTags>
	Array(const Array_Slice<Dimension, value_type, allocator_t, SliceTags...>& expr_t)
	: Allocator(BC::allocator_traits<Allocator>::select_on_container_copy_construction(
			expr_t.get_allocator())),
	  stream_t(expr_t.get_stream())
	{
		this->as_shape() = Shape<Dimension>(expr_t.inner_shape());
		this->array = this->allocate(this->size());
		greedy_evaluate(this->internal(), expr_t.internal(), this->get_stream());
	}

public:
    void internal_move(Array&& array) {
    		if (BC::allocator_traits<Allocator>::is_always_equal::value ||
    			array.get_allocator() == this->get_allocator()) {
			std::swap(this->array, array.array);
			this->as_shape() = array.as_shape();

	    	if (BC::allocator_traits<Allocator>::propagate_on_container_move_assignment::value) {
				static_cast<Allocator&>(*this) = static_cast<Allocator&&>(array);
			}
    	} else {
    		get_allocator().deallocate(this->array, this->size());
			this->as_shape() = array.as_shape();
			this->array = get_allocator().allocate(this->size());
			greedy_evaluate(this->internal(), array.internal(), this->get_stream());
    	}
    }

    void deallocate() {
       Allocator::deallocate(this->array, this->size());
       this->array = nullptr;
	}

};

template<class ValueType, int dims, class Stream>
auto make_temporary_kernel_array(Shape<dims> shape, Stream stream) {
	static_assert(dims >= 1, "make_temporary_tensor_array: assumes dimension is 1 or greater");
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

template<int Dimension, class ValueType, class SystemTag, class... Tags, class=std::enable_if_t<BC::meta::sequence_contains_v<BC_Temporary, Tags...>>>
void destroy_temporary_kernel_array(Kernel_Array<Dimension, ValueType, SystemTag, Tags...> temporary, BC::Stream<SystemTag> stream) {
	stream.template get_allocator_rebound<ValueType>().deallocate(temporary.array, temporary.size());
}


template<class Shape, class Allocator>
auto make_tensor_array(Shape shape, Allocator alloc) {
	return Array<Shape::tensor_dimension, typename Allocator::value_type, Allocator>(shape, alloc);
}


} //ns BC
} //ns exprs
} //ns tensors



#endif /* SHAPE_H_ */
