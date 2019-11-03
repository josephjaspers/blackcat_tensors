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
		BC::traits::sequence_contains_v<noncontinuous_memory_tag, Tags...> ?
				tensor_dimension : 1;

	using value_type = ValueType;
	using system_tag = SystemTag;
	using shape_type = Shape;

protected:

	value_type* m_data = nullptr;

public:

	Kernel_Array()=default;

	Kernel_Array(shape_type shape, value_type* ptr):
		shape_type(shape), m_data(ptr) {};

	template<class AllocatorType>
	Kernel_Array(shape_type shape, AllocatorType allocator):
		shape_type(shape),
		m_data(allocator.allocate(this->size())) {};

	BCINLINE
	value_type* data() const {
		return m_data;
	}

	BCINLINE
	shape_type get_shape() const {
		return static_cast<const shape_type&>(*this);
	}

	BCINLINE
	const auto& operator [](BC::size_t index) const {
		return m_data[this->coefficientwise_dims_to_index(index)];
	}

	BCINLINE
	auto& operator [](BC::size_t index) {
		return m_data[this->coefficientwise_dims_to_index(index)];
	}

	template<class ... Integers> BCINLINE
	const auto& operator ()(Integers ... ints) const {
		return m_data[this->dims_to_index(ints...)];
	}

	template<class ... Integers> BCINLINE
	auto& operator ()(Integers ... ints) {
		return m_data[this->dims_to_index(ints...)];
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


	template<class Allocator> BCHOT
	void deallocate(Allocator allocator) {
		allocator.deallocate(data(), this->size());
	}


	//TODO remove
	void deallocate() const {};

	template<class Allocator> BCHOT
	void reset(Allocator allocator) {
		deallocate(allocator);
		static_cast<shape_type&>(*this) = shape_type();
	}
};


template<int N, class Allocator, class... Tags>
auto make_kernel_array(Shape<N> shape, Allocator allocator, Tags...) {
	using system_tag = typename BC::allocator_traits<Allocator>::system_tag;
	using value_type = typename BC::allocator_traits<Allocator>::value_type;
	return Kernel_Array<Shape<N>, value_type, system_tag, Tags...>(shape, allocator);
}


}
}
}


#endif /* BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_ */
