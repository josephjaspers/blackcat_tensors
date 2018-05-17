/*
 * Core_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "BlackCat_Internal_Definitions.h"

namespace BC {

/*
 * Core_Interface is a common interface amongst all tensor_core subclasses,
 */

template<class> class Tensor_Reshape;
template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class> class Tensor_Row;
template<class> class Tensor_Chunk;

template<class derived, int DIMENSION,
template<class> class 	_Tensor_Reshape = Tensor_Reshape,
template<class> class 	_Tensor_Slice 	= Tensor_Slice,
template<class> class 	_Tensor_Row 	= Tensor_Row,
template<class> class 	_Tensor_Scalar 	= Tensor_Scalar,
template<class> class   _Tensor_Chunk	= Tensor_Chunk>			//Nested implementation type
struct Core_Base : expression_base<Core_Base<derived,DIMENSION>> {

	__BCinline__ static constexpr bool ASSIGNABLE() { return true; }
	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int PARENT_DIMS() { return DIMS(); }

	__BCinline__ static constexpr int CONTINUOUS() { return 0; }
	__BCinline__ static constexpr int LAST() { return DIMENSION - 1; }

	using self = derived;

	using slice_type = std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using row_type = std::conditional_t<DIMS() == 0, self, _Tensor_Row<self>>;
	using scalar_type = std::conditional_t<DIMS() == 0, self, _Tensor_Scalar<self>>;

private:
	__BCinline__ const derived& base() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& base() 		 { return static_cast<	    derived&>(*this); }

public:
	operator 	   auto()       { return base().getIterator(); }
	operator const auto() const { return base().getIterator(); }

	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; }
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; }
	template<class... integers> __BCinline__ 	   auto& operator () (integers... ints) 	  {
//FIXME, error in the reading function // static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[this->scal_index(ints...)]; }
	template<class... integers> __BCinline__ const auto& operator () (integers... ints) const {
//FIXME, error in the reading function // static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[this->scal_index(ints...)]; }

	__BCinline__ const auto innerShape() const { return base().innerShape(); }
	__BCinline__ const auto outerShape() const { return base().outerShape(); }

	__BCinline__ const auto slice(int i) const { return slice_type(&(base().getIterator()[slice_index(i)]),base()); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&(base().getIterator()[slice_index(i)]),base()); }

	__BCinline__ const auto scalar(int i) const { return scalar_type(&base().getIterator()[i], base()); }
	__BCinline__	   auto scalar(int i) 	    { return scalar_type(&base().getIterator()[i], base()); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return row_type(&base().getIterator()[i], base()); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return row_type(&base().getIterator()[i], base()); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	template<class ... integers> __BCinline__ auto chunk(integers ... location_indices) {
		return [&, location_indices...](auto... shape_dimension) {
			auto* array = &(this->base().getIterator()[this->scal_index(location_indices...)]);
			return typename _Tensor_Chunk<derived>::template implementation<sizeof...(shape_dimension)>(array, this->base(), shape_dimension...);
		};
	}

	template<class ... integers> __BCinline__ const auto chunk(integers ... location_indices) const {
		return [&, location_indices...](auto... shape_dimension) {
			auto* array = &(this->base().getIterator()[this->scal_index(location_indices...)]);
			return typename _Tensor_Chunk<derived>::template implementation<sizeof...(shape_dimension)>(array, this->base(), shape_dimension...);
		};
	}

	template<class ... integers> __BCinline__
	const auto reshape(integers ... ints) const {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		using tensor_type = tensor_of_t<sizeof...(integers), self, _mathlib<derived>>;
		return Tensor_Reshape<tensor_type>(base(), ints...);
	}

	template<class... integers> __BCinline__
	auto reshape(integers... ints) {
		static constexpr bool int_seq = MTF::is_integer_sequence<integers...>;
		static_assert(int_seq, "MUST BE INTEGER LIST");

		using tensor_type = tensor_of_t<sizeof...(integers), self, _mathlib<derived>>;
		return Tensor_Reshape<tensor_type>(base(), ints...);
	}


	//------------------------------------------Implementation Details---------------------------------------//

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return base().outerShape()[LAST() - 1] * i;
	}


};

}


#endif /* TENSOR_CORE_INTERFACE_H_ */
