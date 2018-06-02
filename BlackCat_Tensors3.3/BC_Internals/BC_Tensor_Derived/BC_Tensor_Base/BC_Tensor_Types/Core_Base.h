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
template<class> class Tensor_Vectorizer;
template<int> class Tensor_Reshape;
template<int> class Tensor_Chunk;

template<class derived, int DIMENSION,
template<class> class 	_Tensor_Slice 	= Tensor_Slice,
template<class> class 	_Tensor_Row 	= Tensor_Row,
template<class> class 	_Tensor_Scalar 	= Tensor_Scalar,
template<class> class 	_Tensor_Vectorizer = Tensor_Vectorizer,
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
	__BCinline__ const auto inner_shape() const { return asDerived().inner_shape(); }
	__BCinline__ const auto outer_shape() const { return asDerived().outer_shape(); }
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
	//currently cuda will not compile these
	template<class ... integers>  auto chunk(int location, integers ... shape_dimensions) {
		static constexpr int tensor_dim =  sizeof...(shape_dimensions);
		return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(&asDerived().getIterator()[location], this->asDerived(), shape_dimensions...);
	}
	template<class ... integers>  const auto chunk(int location, integers ... shape_dimensions) const {
		static constexpr int tensor_dim =  sizeof...(shape_dimensions);
		return typename _Tensor_Chunk<tensor_dim>::template implementation<derived>(&asDerived().getIterator()[location], this->asDerived(), shape_dimensions...);
	}



	template<class ... integers>
	auto reshape(integers... ints) {
		static constexpr int tensor_dim =  sizeof...(ints);
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), ints...);
	}

	template<int dim>
	auto reshape(Shape<dim> shape) {
		return typename _Tensor_Reshape<dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), shape);
	}
	template<class ... integers>
	const auto reshape(integers... ints) const {
		static constexpr int tensor_dim =  sizeof...(ints);
		return typename _Tensor_Reshape<tensor_dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), ints...);
	}

	template<int dim>
	const auto reshape(Shape<dim> shape) const  {
		return typename _Tensor_Reshape<dim>::template implementation<derived>(asDerived().getIterator(), this->asDerived(), shape);
	}


//#endif

	//------------------------------------------Implementation Details---------------------------------------//

	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return asDerived().outer_shape()[DIMENSION - 2] * i;
	}


};

template<class T, class voider = void> struct isCore { static constexpr bool conditional = false; };
template<class T> struct isCore<T, std::enable_if_t<std::is_same<decltype(T::DIMS()),int>::value>> { static constexpr bool conditional = std::is_base_of<Tensor_Core_Base<T, T::DIMS()>, T>::value; };

template<class T> static constexpr bool isCore_b = isCore<T>::conditional;

}


#endif /* TENSOR_CORE_INTERFACE_H_ */
