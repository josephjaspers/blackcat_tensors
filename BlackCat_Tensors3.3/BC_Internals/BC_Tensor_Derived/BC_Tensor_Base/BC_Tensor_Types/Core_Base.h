/*
 * Core_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Internal_Shape.h"

namespace BC {

/*
 * Core_Interface is a common interface amongst all tensor_core subclasses,
 */

template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class> class Tensor_Row;
template<int> class Tensor_Reshape;
template<int> class Tensor_Chunk;

template<class derived, int DIMENSION,
template<class> class 	_Tensor_Slice 	= Tensor_Slice,
template<class> class 	_Tensor_Row 	= Tensor_Row,
template<class> class 	_Tensor_Scalar 	= Tensor_Scalar,
template<int> class     _Tensor_Chunk	= Tensor_Chunk,			//Nested implementation type
template<int> class 	_Tensor_Reshape = Tensor_Reshape>		//Nested implementation type

struct Tensor_Core_Base : expression_base<Tensor_Core_Base<derived,DIMENSION>> {

	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int ITERATOR() { return 0; }
	__BCinline__ static constexpr bool ASSIGNABLE() { return true; }

	using self = derived;
	using slice_type = std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using row_type = std::conditional_t<DIMS() == 0, self, _Tensor_Row<self>>;
	using scalar_type = std::conditional_t<DIMS() == 0, self, _Tensor_Scalar<self>>;

private:
	__BCinline__ const derived& asDerived() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& asDerived() 	  { return static_cast<	     derived&>(*this); }

public:
	operator 	   auto()       { return asDerived().getIterator(); }
	operator const auto() const { return asDerived().getIterator(); }

	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? asDerived().getIterator()[0] : asDerived().getIterator()[index]; }
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? asDerived().getIterator()[0] : asDerived().getIterator()[index]; }
	__BCinline__ const auto innerShape() const { return asDerived().innerShape(); }
	__BCinline__ const auto outerShape() const { return asDerived().outerShape(); }
	__BCinline__ const auto slice(int i) const { return slice_type(&(asDerived().getIterator()[slice_index(i)]),asDerived()); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&(asDerived().getIterator()[slice_index(i)]),asDerived()); }
	__BCinline__ const auto scalar(int i) const { return scalar_type(&asDerived().getIterator()[i], asDerived()); }
	__BCinline__	   auto scalar(int i) 	    { return scalar_type(&asDerived().getIterator()[i], asDerived()); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return row_type(&asDerived().getIterator()[i], asDerived()); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return row_type(&asDerived().getIterator()[i], asDerived()); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	template<class... integers> __BCinline__ 	   auto& operator () (integers... ints) 	  {
		return DIMS() == 0 ? asDerived().getIterator()[0] : asDerived().getIterator()[this->dims_to_index(ints...)]; }
	template<class... integers> __BCinline__ const auto& operator () (integers... ints) const {
		return DIMS() == 0 ? asDerived().getIterator()[0] : asDerived().getIterator()[this->dims_to_index(ints...)]; }

	//------------------------------------------Curried Reshapers ---------------------------------------//

	template<class ... integers> __BCinline__ auto chunk(integers ... location_indices) {
		return [&](auto maybe_shape,  auto... shape_dimension) {
			auto* array = &(this->asDerived().getIterator()[this->dims_to_index_reverse(location_indices...)]);
			static constexpr int tensor_dim = is_shape<decltype(maybe_shape)> ? LENGTH<decltype(maybe_shape)> : sizeof...(shape_dimension) + 1;
			return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(array, this->asDerived(), maybe_shape, shape_dimension...);
		};
	}

	template<class ... integers> __BCinline__ const auto chunk(integers ... location_indices) const {
		return [&](auto maybe_shape, auto... shape_dimension) {
			auto* array = &(this->asDerived().getIterator()[this->dims_to_index_reverse(location_indices...)]);
			static constexpr int tensor_dim = is_shape<decltype(maybe_shape)> ? LENGTH<decltype(maybe_shape)> : sizeof...(shape_dimension) + 1;
			return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(array, this->asDerived(), maybe_shape, shape_dimension...);
		};
	}

	template<class maybe_shape, class ... integers> __BCinline__
	const auto reshape(maybe_shape sh, integers ... ints) const {
		static constexpr int tensor_dim = is_shape<maybe_shape> ? LENGTH<maybe_shape> : sizeof...(ints) + 1;
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), sh, ints...);
	}

	template<class maybe_shape, class ... integers> __BCinline__
	auto reshape(maybe_shape sh, integers... ints) {
		static constexpr int tensor_dim = is_shape<maybe_shape> ? LENGTH<maybe_shape> : sizeof...(ints) + 1;
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), sh, ints...);
	}


	//------------------------------------------Implementation Details---------------------------------------//

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return asDerived().outerShape()[DIMENSION - 2] * i;
	}


};

}


#endif /* TENSOR_CORE_INTERFACE_H_ */
