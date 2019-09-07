/*
 * Array_Kernel_Array.h
 *
 *  Created on: Sep 7, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_
#define BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_

#include "Expression_Template_Base.h"
#include "Shape.h"

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
		  Shape<Dimension>,
		  public Tags... {

    using value_type = ValueType;
    using system_tag = SystemTag;
    using pointer_type = value_type*;
    using shape_type = Shape<Dimension>;
    using self_type  = Kernel_Array<Dimension, ValueType, SystemTag, Tags...>;

	static constexpr bool self_is_view = BC::traits::sequence_contains_v<BC_View, Tags...>;
	static constexpr bool is_continuous = ! BC::traits::sequence_contains_v<BC_Noncontinuous, Tags...>;

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
    Kernel_Array(shape_type shape, AllocatorType allocator)
    	: shape_type(shape), array(allocator.allocate(this->size())) {};

    BCINLINE pointer_type memptr() const { return array; }
    BCINLINE const shape_type& get_shape() const { return static_cast<const shape_type&>(*this); }

    BCINLINE const auto& operator [](int index) const {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else if (!expression_traits<self_type>::is_continuous::value && tensor_dimension==1) {
    		return array[this->leading_dimension(0) * index];
    	} else {
    		return array[index];
    	}
    }

    BCINLINE auto& operator [](int index) {
    	if (tensor_dimension==0) {
    		return array[0];
    	} else if (!expression_traits<self_type>::is_continuous::value && tensor_dimension==1) {
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



}
}
}



#endif /* BLACKCATTENSOR_TENSORS_EXPRESSION_TEMPLATES_ARRAY_KERNEL_ARRAY_H_ */
