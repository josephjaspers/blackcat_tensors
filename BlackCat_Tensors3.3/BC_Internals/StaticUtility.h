/*
 * StaticUtility.h
 *
 *  Created on: Mar 31, 2018
 *      Author: joseph
 */

#ifndef STATICUTILITY_H_
#define STATICUTILITY_H_

#include "BC_Core/TensorBase.h"

namespace BC {


	//returns a lightweight list class in which you may use the bracket operator[] for indexing, does not store the size of the array
//template<class var>
//static auto shapeOf(const var& v) {
//	return reference_array<const var&>(v);
//}
//
//	template<class d> static std::enable_if_t<d::DIMS() <= 1, 	  d&> flatten(	    TensorBase<d>& tensor) { return tensor; }
//	template<class d> static std::enable_if_t<d::DIMS() <= 1, const d&> flatten(const TensorBase<d>& tensor) { return tensor; }
//
//	template<class d> static std::enable_if_t<(d::DIMS() > 1),
//			typename base<1>::template type <_scalar<d>, _mathlib<d>>
//			>  flatten(TensorBase<d>& tensor) {
//
//			 Vector<Tensor_Core<Vector<_scalar<d>, _mathlib<d>>>, _mathlib<d>> flat(std::true_type());
//			 flat.black_cat_array.array = tensor.black_cat_array.array;
//			 flat.black_cat_array.is = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//			 flat.black_cat_array.os = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//
//	}
//
//	template<class d> static std::enable_if_t<(d::DIMS() > 1),
//			const typename base<1>::template type <_scalar<d>, _mathlib<d>>
//			>  flatten(const TensorBase<d>& tensor) {
//
//			 Vector<Tensor_Core<Vector<_scalar<d>, _mathlib<d>>>, _mathlib<d>> flat(std::true_type());
//			 flat.black_cat_array.array = tensor.black_cat_array.array;
//			 flat.black_cat_array.is = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//			 flat.black_cat_array.os = &(tensor.black_cat_array.os[d::DIMS() - 1]);
//
//	}
}


#endif /* STATICUTILITY_H_ */
