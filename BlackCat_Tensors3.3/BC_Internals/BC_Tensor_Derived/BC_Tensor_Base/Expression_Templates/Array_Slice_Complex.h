/*
 * Array_Complex_Slice.h
 *
 *  Created on: Jul 29, 2018
 *      Author: joseph
 */

#ifndef ARRAY_COMPLEX_SLICE_H_
#define ARRAY_COMPLEX_SLICE_H_

#include "Array_Base.h"

namespace BC {
namespace internal {

template<int axis>
struct Array_Slice_Complex {

	static_assert(axis != 0, "COMPLEX SLICE IS NOT LEGAL FOR ROW_SLICES");

	template<class PARENT>
	struct implementation : Tensor_Array_Base<implementation<PARENT>, MTF::max(PARENT::DIMS() - 1, 0)> {
		using scalar_type = _scalar<PARENT>;

		__BCinline__ static constexpr int DIMS() { return MTF::max(PARENT::DIMS() - 1, 0); }
		__BCinline__ static constexpr int ITERATOR() { return MTF::max(PARENT::ITERATOR() - 1, DIMS()); }

		__BCinline__ operator const PARENT() const	{ return parent; }

		const PARENT parent;
		scalar_type* array_slice;

		__BCinline__ implementation(const scalar_type* array, PARENT parent_) : array_slice(const_cast<scalar_type*>(array)), parent(parent_) {}

		__BCinline__ const auto inner_shape() const {
			return l_array<DIMS()>([&](int i) {
				return i < axis ? parent.dimension(i) : parent.dimension(i + 1);
			});
		}
		__BCinline__ const auto outer_shape() const {
			return l_array<DIMS()>([&](int i) {
				return i < axis ? parent.leading_dimension(i) : parent.leading_dimension(i + 1);
			});
		}
		__BCinline__ int size() const {
			std::cout << " pa sz = " << parent.size() << std::endl;
			std::cout << "bs [axis ] = " << parent.dimension(axis) << std::endl;
			return parent.size() / parent.dimension(axis); }
		__BCinline__ int rows() const { return inner_shape()[0]; }
		__BCinline__ int cols() const { return  inner_shape()[1]; }
		__BCinline__ int dimension(int i) const { return inner_shape()[i]; }
		__BCinline__ int outer_dimension() const { return inner_shape()[DIMS() - 1]; }
		__BCinline__ int leading_dimension(int i) const { return outer_shape()[i]; }

		__BCinline__ const scalar_type* memptr() const { return array_slice; }
		__BCinline__	   scalar_type* memptr()   	   { return array_slice; }

	};

};


}
}


#endif /* ARRAY_COMPLEX_SLICE_H_ */
