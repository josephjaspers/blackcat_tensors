/*
 * Array_Kernel_Array.h
 *
 *  Created on: Sep 7, 2019
 *	  Author: joseph
 */

#ifndef BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_
#define BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_

#include "Expression_Template_Base.h"

namespace BC {
namespace tensors {
namespace exprs {


/*
 * 	Array is a class that inherits from Kernel_Array and the Allocator type.
 *  The Array class is used to initialize and destruct the Kernel_Array object.
 *
 *  Array and Kernel_Array are two tightly coupled classes as
 *  expression-templates must be trivially copyable
 *  (to pass them to CUDA functions).
 *
 *  Separating these two enables the usage of non-trivially copyable allocators
 *  as well as the ability to define
 *  non-default move and copy assignments/constructors.
 *
 *  The Kernel_Array class should never be instantiated normally.
 *  It should only be accessed by instantiating an instance of the Array class,
 *  and calling 'my_array_object.internal()' to query it.
 *
 *  Additionally this design pattern (replicated in Array_View)
 *  enables expression-templates to define additional members
 *  that we do not want to pass to the GPU.
 *  (As they my be non-essential to the computation).
 *
 */
template<class Shape, class ValueType, class SystemTag, class... Tags>
struct Kernel_Array:
		Kernel_Array_Base<Kernel_Array<Shape, ValueType, SystemTag, Tags...>>,
		Shape,
		public Tags... {

	static constexpr int tensor_dimension = Shape::tensor_dimension;
	static constexpr int tensor_iterator_dimension =
		BC::traits::sequence_contains_v<BC_Noncontinuous, Tags...> ?
				tensor_dimension : 1;

	using value_type = ValueType;
	using system_tag = SystemTag;
	using pointer_type = value_type*;
	using shape_type = Shape;
	using self_type  = Kernel_Array<Shape, ValueType, SystemTag, Tags...>;

private:

	pointer_type array = nullptr;

protected:

	BCINLINE
	pointer_type& memptr_ref() {
		return array;
	}

	BCINLINE
	shape_type& get_shape_ref() {
		return static_cast<shape_type&>(*this);
	}

public:

	Kernel_Array()=default;
	Kernel_Array(const Kernel_Array&)=default;
	Kernel_Array(Kernel_Array&&)=default;
	Kernel_Array(shape_type shape, pointer_type ptr):
		shape_type(shape), array(ptr) {};

	template<class AllocatorType>
	Kernel_Array(shape_type shape, AllocatorType allocator):
		shape_type(shape), array(allocator.allocate(this->size())) {};

	BCINLINE
	pointer_type memptr() const {
		return array;
	}

	BCINLINE
	shape_type get_shape() const {
		return static_cast<const shape_type&>(*this);
	}

	BCINLINE
	const auto& operator [](BC::size_t index) const {
		return array[this->coefficientwise_dims_to_index(index)];
	}

	BCINLINE
	auto& operator [](BC::size_t index) {
		return array[this->coefficientwise_dims_to_index(index)];
	}

	template<class ... Integers> BCINLINE
	const auto& operator ()(Integers ... ints) const {
		return array[this->dims_to_index(ints...)];
	}

	template<class ... Integers> BCINLINE
	auto& operator ()(Integers ... ints) {
		return array[this->dims_to_index(ints...)];
	}

	BCINLINE
	auto slice_ptr_index(int i) const {
		if (tensor_dimension == 0)
			return 0;
		else if (tensor_dimension == 1)
			return i;
		else
			return this->leading_dimension(Shape::tensor_dimension - 1) * i;
	}

};


template<class ValueType, int Dimension, class Stream>
auto make_temporary_kernel_array(Shape<Dimension> shape, Stream stream) {
	using system_tag = typename Stream::system_tag;
	using Array = Kernel_Array<Shape<Dimension>, ValueType, system_tag, BC_Temporary>;
	return Array(shape, stream.template get_allocator_rebound<ValueType>().allocate(shape.size()));
}

template<class ValueType, class Stream>
auto make_temporary_kernel_scalar(Stream stream) {
	using system_tag = typename Stream::system_tag;
	using Array = Kernel_Array<Shape<0>, ValueType, system_tag, BC_Temporary>;
	return Array(BC::Shape<0>(), stream.template get_allocator_rebound<ValueType>().allocate(1));
}

template<
	int Dimension,
	class ValueType,
	class Stream,
	class... Tags,
	class=std::enable_if_t<
		BC::traits::sequence_contains_v<BC_Temporary, Tags...>>>
void destroy_temporary_kernel_array(
		Kernel_Array<Shape<Dimension>, ValueType, typename Stream::system_tag, Tags...> temporary, Stream stream) {
	stream.template get_allocator_rebound<ValueType>().deallocate(temporary.memptr(), temporary.size());
}


}
}
}


#endif /* BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_ */
