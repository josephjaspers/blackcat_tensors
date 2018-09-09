
/*
 * Array_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Internal_Type_Interface.h"
#include "Shape.h"

namespace BC {
namespace internal {

/*
 * Array_Interface is a common interface amongst all tensor_core subclasses,
 */


//required forward decls
template<class> class Array_Slice;
template<class> class Array_Slice_Range;

template<class> class Array_Scalar;
template<class> class Array_Transpose;
template<class> class Array_Row;
template<int> class Array_Reshape;
template<int> class Array_Chunk;


//Many template params
template<class derived, int DIMENSION,
template<class> class 	_Tensor_Slice 	= Array_Slice,
template<class> class 	_Tensor_Range	= Array_Slice_Range,
template<class> class	_Tensor_Row 	= Array_Row,
template<class> class 	_Tensor_Scalar 	= Array_Scalar,
template<int> class     _Tensor_Chunk	= Array_Chunk,			//Nested implementation type
template<int> class 	_Tensor_Reshape = Array_Reshape>		//Nested implementation type

struct Array_Base : BC_internal_interface<derived>, BC_Array {

	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }

	using self 		= derived;
	using slice_t 	= std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using scalar_t 	= std::conditional_t<DIMS() == 0, self, _Tensor_Scalar<self>>;
	using row_t =  _Tensor_Row<self>;
	using slice_range_t = _Tensor_Range<self>;
	template<int dimension> using reshape_t = typename _Tensor_Reshape<dimension>::template implementation<derived>;
	template<int dimension> using chunk_t 	= typename _Tensor_Chunk<dimension>::template implementation<derived>;

private:

	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

public:

	operator 	   auto()       { return as_derived().memptr(); }
	operator const auto() const { return as_derived().memptr(); }

	__BCinline__ const auto& operator [](int index) const {
		return as_derived().memptr()[index];
	}
	__BCinline__ auto& operator [](int index) {
		return as_derived().memptr()[index];
	}

	template<class ... integers>
	__BCinline__ auto& operator ()(integers ... ints) {
		return as_derived()	[this->dims_to_index(ints...)];
	}
	template<class ... integers>
	__BCinline__ const auto& operator ()(integers ... ints) const {
		return as_derived()[this->dims_to_index(ints...)];
	}

	//internal_views---------------------------------------------------------------------------------------------------------------

	__BCinline__ const auto _slice(int i) const {
		//change to if constexpr once NVCC supports it
		struct ret_scalar { __BCinline__ static auto impl(const Array_Base& self, int i) { return self._scalar(i); } };
		struct ret_slice  { __BCinline__ static auto impl(const Array_Base& self, int i) {
			return slice_t(self.slice_ptr(i), self.as_derived()); } };

		using xslice_t = std::conditional_t<DIMS() == 1, ret_scalar,
				std::conditional_t<(DIMS() > 1), ret_slice, void>>;

		return xslice_t::impl(*this);
	}
	__BCinline__ auto _slice(int i) {
		//change to if constexpr once NVCC supports it
		struct ret_scalar { __BCinline__ static auto impl(const Array_Base& self, int i) { return self._scalar(i); } };
		struct ret_slice  { __BCinline__ static auto impl(const Array_Base& self, int i) {
			return slice_t(self.slice_ptr(i), self.as_derived()); } };

		using xslice_t = std::conditional_t<DIMS() == 1, ret_scalar,
				std::conditional_t<(DIMS() > 1), ret_slice, void>>;

		return xslice_t::impl(*this, i);
	}

	__BCinline__ const auto _slice_range(int from, int to) const {
		return slice_range_t(slice_ptr(from), *this, to - from);
	}
	__BCinline__ auto _slice_range(int from, int to) {
		return slice_range_t(slice_ptr(from), *this, to - from);
	}


	__BCinline__ const auto _row(int i) const {
		return row_t(&as_derived()[i], as_derived());
	}
	__BCinline__  auto _row(int i)  {
		return row_t(&as_derived()[i], as_derived());
	}


	__BCinline__ const auto _scalar(int i) const {
		static_assert(derived::ITERATOR() == 0 || derived::ITERATOR() == 1, "SCALAR_ACCESS IS NOT ALLOWED FOR NON CONTINUOUS TENSORS");
		return scalar_t(&as_derived()[i]);
	}
	__BCinline__ auto _scalar(int i) {
		static_assert(derived::ITERATOR() == 0 || derived::ITERATOR() == 1, "SCALAR_ACCESS IS NOT ALLOWED FOR NON CONTINUOUS TENSORS");
		return scalar_t(&as_derived()[i]);
	}


	//------------------------------------------Curried Reshapers ---------------------------------------//

	template<class ... integers> auto _chunk(int location, integers ... shape_dimensions) {
		return chunk_t<sizeof...(integers)>(&as_derived()[location], this->as_derived(), shape_dimensions...);
	}
	template<class ... integers>  const auto _chunk(int location, integers ... shape_dimensions) const {
		return chunk_t<sizeof...(integers)>(&as_derived()[location], this->as_derived(), shape_dimensions...);
	}

	template<class ... integers>
	auto _reshape(integers... ints) {
		return reshape_t<sizeof...(integers)>(as_derived(), this->as_derived(), ints...);
	}

	template<int dim>
	auto _reshape(Shape<dim> shape) {
		return reshape_t<dim>(as_derived(), this->as_derived(), shape);
	}
	template<class ... integers>
	const auto _reshape(integers... ints) const {
		return reshape_t<sizeof...(integers)>(as_derived(), this->as_derived(), ints...);
	}

	template<int dim>
	const auto _reshape(Shape<dim> shape) const  {
		return reshape_t<dim>(as_derived(), this->as_derived(), shape);
	}



	void destroy() {}
	//------------------------------------------Implementation Details---------------------------------------//

	__BCinline__
	auto slice_ptr(int i) const {
		if (DIMS() == 0)
			return &as_derived()[0];
		else if (DIMS() == 1)
			return &as_derived()[i];
		else
			return &as_derived()[as_derived().leading_dimension(DIMENSION - 2) * i];
	}



	//---------------------------------------------------UTILITY/IMPLEMENTATION METHODS------------------------------------------------------------//


	template<class... integers> __BCinline__
	int dims_to_index(integers... ints) const {
		return dims_to_index(BC::make_array(ints...));
	}
	template<class... integers> __BCinline__
	int dims_to_index_reverse(integers... ints) const {
		return dims_to_index_reverse(BC::make_array(ints...));
	}

//	__BCinline__ int dims_to_index(int index) const {
//		return index * as_derived().leading_dimension(0);
//	}
//	__BCinline__ int dims_to_index_reverse(int index) const {
//		return index * as_derived().leading_dimension(0);
//	}

	template<int D> __BCinline__ int dims_to_index(BC::array<D, int> var) const {
		int index = var[0];
		for(int i = 1; i < DIMS(); ++i) {
			index += this->as_derived().leading_dimension(i - 1) * var[i];
		}
		return index;
	}
	template<int D> __BCinline__ int dims_to_index_reverse(BC::array<D, int> var) const {
		static_assert(D >= DIMS(), "converting array_to dimension must have at least as many indices as the tensor");

		int index = var[DIMS() - 1];
		for(int i = 0; i < DIMS() - 1; ++i) {
			index += this->as_derived().leading_dimension(i) * var[DIMS() - i - 2];
		}
		return index;
	}
};
}

template<class T> static constexpr bool is_array() { return std::is_base_of<internal::Array_Base<T, T::DIMS()>, T>::value; };

template<class T>
struct BC_array_copy_assignable_overrider<T, std::enable_if_t<is_array<T>()>> {
	static constexpr bool boolean = true;
};


}


#endif /* TENSOR_CORE_INTERFACE_H_ */
