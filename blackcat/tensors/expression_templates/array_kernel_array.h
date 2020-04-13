/*
 * Array_Kernel_Array.h
 *
 *  Created on: Sep 7, 2019
 *	  Author: joseph
 */

#ifndef BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_
#define BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_

#include "expression_template_base.h"

namespace bc {
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
 *  and calling 'my_array_object.expression_template()' to query it.
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

	static constexpr int tensor_dim = Shape::tensor_dim;
	static constexpr int tensor_iterator_dim =
		bc::traits::sequence_contains_v<noncontinuous_memory_tag, Tags...> ?
				tensor_dim : 1;

	using value_type = ValueType;
	using system_tag = SystemTag;
	using shape_type = Shape;

private:

using return_type = std::conditional_t<
		bc::traits::sequence_contains_v<BC_Const_View, Tags...>,
		const value_type,
		value_type>;

protected:

	value_type* m_data = nullptr;

public:

	BCINLINE
	Kernel_Array() {}

	BCINLINE
	Kernel_Array(shape_type shape, value_type* ptr):
		shape_type(shape), m_data(ptr) {};

	template<
		class AllocatorType,
		class=std::enable_if_t<
				bc::traits::true_v<
						decltype(std::declval<AllocatorType>().allocate(0))>>>
	Kernel_Array(shape_type shape, AllocatorType allocator):
		shape_type(shape),
		m_data(allocator.allocate(this->size())) {};

public:


	BCINLINE
	value_type* data() const {
		return m_data;
	}

	BCINLINE
	shape_type get_shape() const {
		return static_cast<const shape_type&>(*this);
	}

	BCINLINE
	const return_type& operator [](bc::size_t index) const {
		return m_data[this->coefficientwise_dims_to_index(index)];
	}

	BCINLINE
	return_type& operator [](bc::size_t index) {
		return m_data[this->coefficientwise_dims_to_index(index)];
	}

	template<class ... Integers> BCINLINE
	const return_type& operator ()(Integers ... ints) const {
		return m_data[this->dims_to_index(ints...)];
	}

	template<class ... Integers> BCINLINE
	return_type& operator ()(Integers ... ints) {
		return m_data[this->dims_to_index(ints...)];
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
	using system_tag = typename bc::allocator_traits<Allocator>::system_tag;
	using value_type = typename bc::allocator_traits<Allocator>::value_type;
	using array_t = Kernel_Array<Shape<N>, value_type, system_tag, Tags...>;
	return array_t(shape, allocator);
}


}
}
}


#endif /* BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_ */
