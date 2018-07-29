
/*
 * Array_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "BlackCat_Internal_Shape.h"

namespace BC {
namespace internal {

/*
 * Array_Interface is a common interface amongst all tensor_core subclasses,
 */

template<class> class Array_Slice;
template<class> class Array_Scalar;
template<class> class Array_Transpose;
template<int> class Array_Reshape;
template<int> class Array_Chunk;

template<class derived, int DIMENSION,
template<class> class 	_Tensor_Slice 	= Array_Slice,
template<class> class 	_Tensor_Scalar 	= Array_Scalar,
template<int> class     _Tensor_Chunk	= Array_Chunk,			//Nested implementation type
template<int> class 	_Tensor_Reshape = Array_Reshape>		//Nested implementation type

struct Tensor_Array_Base : expression_base<derived>, BC_Array {

	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	using self 		= derived;
	using slice_t 	= std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using scalar_t 	= std::conditional_t<DIMS() == 0, self, _Tensor_Scalar<self>>;

private:

	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

public:

	operator 	   auto()       { return as_derived().memptr(); }
	operator const auto() const { return as_derived().memptr(); }

	__BCinline__ const auto& operator [] (int index) const { return as_derived().memptr()[index]; }
	__BCinline__ 	   auto& operator [] (int index) 	   { return as_derived().memptr()[index]; }
	__BCinline__ const auto slice		(int i) const { return slice_t (&as_derived()[slice_index(i)], as_derived()); }
	__BCinline__	   auto slice		(int i) 	  { return slice_t (&as_derived()[slice_index(i)], as_derived()); }
	__BCinline__ const auto scalar		(int i) const { return scalar_t(&as_derived()[i], as_derived()); }
	__BCinline__	   auto scalar		(int i) 	  { return scalar_t(&as_derived()[i], as_derived()); }
public:
	template<class... integers> __BCinline__ 	   auto& operator () (integers... ints) 	  {
		return DIMS() == 0 ? as_derived()[0] : as_derived()[this->dims_to_index(ints...)];
	}
	template<class... integers> __BCinline__ const auto& operator () (integers... ints) const {
		return DIMS() == 0 ? as_derived()[0] : as_derived()[this->dims_to_index(ints...)];
	}

	//------------------------------------------Curried Reshapers ---------------------------------------//
	//currently cuda will not compile these
	template<class ... integers>  auto chunk(int location, integers ... shape_dimensions) {
		static constexpr int tensor_dim =  sizeof...(shape_dimensions);
		return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(&as_derived()[location], this->as_derived(), shape_dimensions...);
	}
	template<class ... integers>  const auto chunk(int location, integers ... shape_dimensions) const {
		static constexpr int tensor_dim =  sizeof...(shape_dimensions);
		return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(&as_derived()[location], this->as_derived(), shape_dimensions...);
	}



	template<class ... integers>
	auto reshape(integers... ints) {
		static constexpr int tensor_dim =  sizeof...(ints);
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(as_derived(), this->as_derived(), ints...);
	}

	template<int dim>
	auto reshape(Shape<dim> shape) {
		return typename _Tensor_Reshape<dim>::template implementation<derived>(as_derived(), this->as_derived(), shape);
	}
	template<class ... integers>
	const auto reshape(integers... ints) const {
		static constexpr int tensor_dim =  sizeof...(ints);
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(as_derived(), this->as_derived(), ints...);
	}

	template<int dim>
	const auto reshape(Shape<dim> shape) const  {
		return typename _Tensor_Reshape<dim>::template implementation<derived>(as_derived(), this->as_derived(), shape);
	}

	//------------------------------------------Implementation Details---------------------------------------//

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return as_derived().leading_dimension(DIMENSION - 2) * i;
	}



	//---------------------------------------------------UTILITY/IMPLEMENTATION METHODS------------------------------------------------------------//


	template<class... integers> __BCinline__
	int dims_to_index(integers... ints) const {
		return dims_to_index(BC::array(ints...));
	}
	template<class... integers> __BCinline__
	int dims_to_index_reverse(integers... ints) const {
		return dims_to_index_reverse(BC::array(ints...));
	}

	template<int D> __BCinline__ int dims_to_index(stack_array<D, int> var) const {
		int index = var[0];
		for(int i = 1; i < DIMS(); ++i) {
			index += this->as_derived().leading_dimension(i - 1) * var[i];
		}
		return index;
	}
	template<int D> __BCinline__ int dims_to_index_reverse(stack_array<D, int> var) const {
		static_assert(D >= DIMS(), "converting array_to dimension must have at least as many indices as the tensor");

		int index = var[DIMS() - 1];
		for(int i = 0; i < DIMS() - 1; ++i) {
			index += this->as_derived().leading_dimension(i) * var[DIMS() - i - 2];
		}
		return index;
	}



};


template<class T, class voider = void> struct isArray_impl { static constexpr bool conditional = false; };
template<class T> struct isArray_impl<T, std::enable_if_t<std::is_same<decltype(T::DIMS()),int>::value>> { static constexpr bool conditional = std::is_base_of<Tensor_Array_Base<T, T::DIMS()>, T>::value; };

template<class T> static constexpr bool isArray() { return isArray_impl<std::decay_t<T>>::conditional; }

}
}


#endif /* TENSOR_CORE_INTERFACE_H_ */
