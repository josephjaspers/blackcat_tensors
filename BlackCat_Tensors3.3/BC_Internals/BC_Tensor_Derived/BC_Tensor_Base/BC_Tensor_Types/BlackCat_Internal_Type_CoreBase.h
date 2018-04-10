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
struct Tensor_Core_Base : Expression_Core_Base<_scalar<derived>, Tensor_Core_Base<derived,DIMENSION>> {

	__BCinline__ static constexpr int DIMS() { return DIMENSION; }
	__BCinline__ static constexpr int LAST() { return DIMENSION - 1; }
	__BCinline__ static constexpr int PRIORITY() { return 0; }

	using self = derived;
	using slice_type = std::conditional_t<DIMS() == 0, self, _Tensor_Slice<self>>;

private:
	__BCinline__ const derived& base() const { return static_cast<const derived&>(*this); }
	__BCinline__ 	   derived& base() 		 { return static_cast<	    derived&>(*this); }

public:
	operator 	   auto()       { return base().getIterator(); }
	operator const auto() const { return base().getIterator(); }

	__BCinline__ 	   auto& operator [] (int index) 	   { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; };
	__BCinline__ const auto& operator [] (int index) const { return DIMS() == 0 ? base().getIterator()[0] : base().getIterator()[index]; };

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

	template<class ... integers>
	const auto reshape(integers ... ints) const {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return Tensor_Reshape<const self, sizeof...(integers)>(base(), ints...);
	}

	template<class... integers>
	auto reshape(integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return Tensor_Reshape< self, sizeof...(integers)>(base(), ints...);
	}

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
