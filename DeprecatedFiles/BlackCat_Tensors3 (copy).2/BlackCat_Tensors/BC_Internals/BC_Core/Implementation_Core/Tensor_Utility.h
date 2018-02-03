/*
 * Tensor_Lv2_Core_impl.h
 *
 *  Created on: Jan 2, 2018
 *      Author: joseph
 */

#ifndef TENSOR_LV2_CORE_IMPL_H_
#define TENSOR_LV2_CORE_IMPL_H_

namespace BC {

template<class T, class deriv, class Mathlib, bool Utility_Function_Supported>
struct Tensor_Utility {};


template<class scalar_type, class deriv, class MATHLIB>
struct Tensor_Utility<scalar_type, deriv, MATHLIB, true> {

/*
 *  Tensor_Base specialization (for primary tensors -- we enable these utility methods)
 */

	deriv& asDerived() {
		return static_cast<deriv&>(*this);
	}
	const deriv& asDerived() const {
		return static_cast<const deriv&>(*this);
	}

	void randomize(scalar_type lb, scalar_type ub) {
		MATHLIB::randomize(asDerived().data(), lb, ub, asDerived().size());
	}
	void fill(scalar_type value) {
		MATHLIB::fill(asDerived().data(), value, asDerived().size());
	}
	void zero() {
		MATHLIB::zero(asDerived().data(), asDerived().size());
	}
	void zeros() {
		MATHLIB::zero(asDerived().data(), asDerived().size());
	}
	void print() const {
		MATHLIB::print(asDerived().data(), asDerived().InnerShape(), asDerived().rank(), 8);
	}
	void print(int precision) const {
		MATHLIB::print(asDerived().data(), asDerived().InnerShape(), asDerived().order(), precision);
	}
};

}



#endif /* TENSOR_LV2_CORE_IMPL_H_ */
