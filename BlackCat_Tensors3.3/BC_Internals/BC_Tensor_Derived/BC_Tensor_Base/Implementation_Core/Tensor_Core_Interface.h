/*
 * Tensor_Core_Interface.h
 *
 *  Created on: Apr 1, 2018
 *      Author: joseph
 */

#ifndef TENSOR_CORE_BASE_H_
#define TENSOR_CORE_BASE_H_

#include "BlackCat_Internal_Definitions.h"

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

	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int LAST() { return DIMENSION - 1; }

	using me   = Tensor_Core_Base<derived, DIMENSION>;
	using self = derived;
	using slice_type = std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;
	using const_slice_type = std::conditional_t<DIMS() == 0, const self, _Tensor_Slice<const self>>;

private:
	__BCinline__ const derived& base() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& base() 		 { return static_cast<	    derived&>(*this); }

//	template<class T = void> __BCinline__ const auto array() const { return base().getIterator(); };
//	template<class T = void> __BCinline__	    auto array() 	   { return base().getIterator(); };

public:
	operator 	   auto()       { return base().getIterator(); }
	operator const auto() const { return base().getIterator(); }


	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; };
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; };

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? outerShape()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? innerShape()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? innerShape()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? base().innerShape()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? outerShape()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? outerShape()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? outerShape()[i] : 1; }

	__BCinline__ const auto innerShape() const { return base().innerShape(); }
	__BCinline__ const auto outerShape() const { return base().outerShape(); }

	__BCinline__ const auto slice(int i) const { return slice_type(&base().getIterator()[slice_index(i)],base()); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&base().getIterator()[slice_index(i)],base()); }

	__BCinline__ const auto scalar(int i) const { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return _Tensor_Scalar<self>(&base().getIterator()[i], base()); }
	__BCinline__	   auto scalar(int i) 	    { static_assert (DIMS() != 0, "SCALAR OF SCALAR NOT DEFINED"); return _Tensor_Scalar<self>(&base().getIterator()[i], base()); }
	__BCinline__ const auto row(int i) const { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return _Tensor_Row<self>(&base().getIterator()[i], base()); }
	__BCinline__	   auto row(int i) 	     { static_assert (DIMS() == 2, "ROW OF NON-MATRIX NOT DEFINED"); return _Tensor_Row<self>(&base().getIterator()[i], base()); }
	__BCinline__ const auto col(int i) const { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }
	__BCinline__	   auto col(int i) 	     { static_assert (DIMS() == 2, "COL OF NON-MATRIX NOT DEFINED"); return slice(i); }

	void destroy() {}; //NEEDS TO BE HERE BUT DO NOT ADD IMPLEMENTATION ACCEPT MAYBE TO TENSOR_CORE

	template<class ... integers>
	const auto reshape(integers ... ints) const { return Tensor_Reshape<const self, sizeof...(integers)>(base(), ints...); }

	template<class... integers>
		  auto reshape(integers... ints) 		{ return Tensor_Reshape<	  self, sizeof...(integers)>(base(), ints...); }

	__BCinline__
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
