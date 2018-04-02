/*
 * Tensor_Core_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_INTERFACE_H_
#define TENSOR_CORE_INTERFACE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Determiners.h"

namespace BC {

/*
 * Tensor_Core_Interface is a common interface amongst all tensor_core subclasses,
 */

template<class, int> class Tensor_Reshape;
template<class> class Tensor_Slice;
template<class> class Tensor_Scalar;
template<class> class Tensor_Row;

template<class derived, int DIMENSION,
template<class,int> class 	_Tensor_Reshape = Tensor_Reshape,
template<class> 	class 	_Tensor_Slice 	= Tensor_Slice,
template<class> 	class 	_Tensor_Row 	= Tensor_Row,
template<class> 	class 	_Tensor_Scalar 	= Tensor_Scalar>
struct Tensor_Core_Base {

	static constexpr int DIMS() { return DIMENSION; }
	static constexpr int LAST() { return DIMENSION - 1; }

	using me   = Tensor_Core_Base<derived, DIMENSION>;
	using self = derived;
	using slice_type = std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using scalar_type = _scalar<self>;
private:
	const derived& base() const { return static_cast<const derived&>(*this); }
		  derived& base() 		{ return static_cast<	   derived&>(*this); }

	__BCinline__ const auto array() const { return base().getIterator(); };
	__BCinline__	   auto array() 	  { return base().getIterator(); };

public:
	operator 	   scalar_type*()       { return array(); }
	operator const scalar_type*() const { return array(); }


	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? array()[0] : array()[index]; };
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? array()[0] : array()[index]; };

	 int dims() const { return DIMS(); }
	 int size() const { return DIMS() > 0 ? outerShape()[LAST()] : 1;    }
	 int rows() const { return DIMS() > 0 ? innerShape()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? innerShape()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? base().innerShape()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? outerShape()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? outerShape()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? outerShape()[i] : 1; }

	__BCinline__ const auto innerShape() const { return base().innerShape(); }
	__BCinline__ const auto outerShape() const { return base().outerShape(); }

	__BCinline__ const auto slice(int i) const { return slice_type(&array()[slice_index(i)],base()); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&array()[slice_index(i)],base()); }

	__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return _Tensor_Scalar<self>(&array()[i], base()); }
	__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return _Tensor_Scalar<self>(&array()[i], base()); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return _Tensor_Row<self>(&array()[i], base()); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return _Tensor_Row<self>(&array()[i], base()); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	template<class ... integers> __BCinline__
	const auto reshape(integers ... ints) const { return Tensor_Reshape<const self, sizeof...(integers)>(base(), ints...); }

	template<class... integers> __BCinline__
		  auto reshape(integers... ints) 		{ return Tensor_Reshape<	  self, sizeof...(integers)>(base(), ints...); }

	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return base().outerShape()[LAST() - 1] * i;
	}
	void printDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << innerShape()[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << outerShape()[i] << "]";
		}
		std::cout << std::endl;
	}
};

}


#endif /* TENSOR_CORE_INTERFACE_H_ */
