/*
 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Parse_Tree_BLAS_Branch_Evaluator.h"
#include "Parse_Tree_Injection_Wrapper.h"
#include "Operations/Binary.h"
#include "Operations/Unary.h"
#include <limits>

#include <iostream>
#include <type_traits>

namespace BC {
namespace internal {

template<class derived>
class expression_base : BC_Type {

	__BCinline__ static constexpr int  DIMS()       { return derived::DIMS(); }
	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

public:

	operator 	   auto&()       { return as_derived(); }
	operator const auto&() const { return as_derived(); }

	__BCinline__ expression_base() { static_assert(std::is_trivially_copy_constructible<derived>::value, "EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE"); }
	__BCinline__ constexpr int  dims() const { return derived::DIMS(); }


	void print_dimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << as_derived().dimension(i) << "]";
		}
		std::cout << std::endl;
	}
	void print_leading_dimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << as_derived().leading_dimension(i) << "]";
		}
		std::cout << std::endl;
	}

	//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//
	void destroy() const {}
	//---------------------------------------------------METHODS THAT NEED TO BE SHADOWED------------------------------------------------------------//
	static constexpr _scalar<derived> BC_NAN = std::numeric_limits<_scalar<derived>>::quiet_NaN();

	__BCinline__ _scalar<derived> operator [] (int index) const { return BC_NAN; }

	template<class... integers>
	__BCinline__ auto operator()(integers... ints) const { return BC_NAN; }


	//-------------------------------------------------tree re-ordering methods---------------------------------------------------------//
};
}
}

#endif /* EXPRESSION_BASE_H_ */
