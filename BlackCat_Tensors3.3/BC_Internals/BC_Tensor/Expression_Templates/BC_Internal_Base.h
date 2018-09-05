/*
 * BC_Internal_Base.h
 *
 *  Created on: Sep 3, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNAL_BASE_H_
#define BC_INTERNAL_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "BLAS_Feature_Detector.h"
#include "Tree_Evaluator_Injector.h"
#include "Operations/Binary.h"
#include "Operations/Unary.h"
#include <iostream>
#include <type_traits>

namespace BC {
namespace internal {

template<class derived>
class BC_internal_interface : BC_Type {

	__BCinline__ static constexpr int  DIMS()       { return derived::DIMS(); }
	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

public:

	operator 	   auto&()       { return as_derived(); }
	operator const auto&() const { return as_derived(); }

	__BCinline__ BC_internal_interface() {
		static_assert(std::is_trivially_copy_constructible<derived>::value, "INTERNAL_TYPES TYPES MUST BE TRIVIALLY COPYABLE");
//		static_assert(std::is_trivially_copyable<derived>::value, "INTERNAL_TYPES MUST BE TRIVIALLY COPYABLE");
		static_assert(!std::is_same<void, typename derived::scalar_t>::value, "INTERNAL_TYPES MUST HAVE A 'using scalar_t = some_Type'");
		static_assert(!std::is_same<void, typename derived::mathlib_t>::value, "INTERNAL_TYPES MUST HAVE A 'using mathlib_t = some_Type'");
		static_assert(!std::is_same<decltype(std::declval<derived>().inner_shape()), void>::value, "INTERNAL_TYPE MUST DEFINE inner_shape()");
		static_assert(!std::is_same<decltype(std::declval<derived>().block_shape()), void>::value, "INTERNAL_TYPE MUST DEFINE block_shape()");
		static_assert(std::is_same<decltype(std::declval<derived>().rows()), int>::value, "INTERNAL_TYPE MUST DEFINE rows()");
		static_assert(std::is_same<decltype(std::declval<derived>().cols()), int>::value, "INTERNAL_TYPE MUST DEFINE cols()");
		static_assert(std::is_same<decltype(std::declval<derived>().dimension(0)), int>::value, "INTERNAL_TYPE MUST DEFINE dimension(int)");
		static_assert(std::is_same<decltype(std::declval<derived>().block_dimension(0)), int>::value, "INTERNAL_TYPE MUST DEFINE block_dimension(int)");

	}

	void destroy() const {}

};

}
}




#endif /* BC_INTERNAL_BASE_H_ */
