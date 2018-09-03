/*
 * BC_Internal_Base.h
 *
 *  Created on: Sep 3, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNAL_BASE_H_
#define BC_INTERNAL_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Parse_Tree_BLAS_Branch_Evaluator.h"
#include "Parse_Tree_Injection_Wrapper.h"
#include "Operations/Binary.h"
#include "Operations/Unary.h"
#include <iostream>
#include <type_traits>

namespace BC {
namespace internal {

template<class derived>
class BC_internal_base : BC_Type {

	__BCinline__ static constexpr int  DIMS()       { return derived::DIMS(); }
	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

public:

	operator 	   auto&()       { return as_derived(); }
	operator const auto&() const { return as_derived(); }

	__BCinline__ BC_internal_base() {
		static_assert(std::is_trivially_copy_constructible<derived>::value, "EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE");
		static_assert(!std::is_same<void, typename derived::scalar_t>::value, "CLASSES DERIVING EXPRESSION_BASE MUST HAVE A 'using scalar_t = some_Type'");
		static_assert(!std::is_same<void, typename derived::mathlib_t>::value, "CLASSES DERIVING EXPRESSION_BASE MUST HAVE A 'using mathlib_t = some_Type'");

	}


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

	void destroy() const {}

};


//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//

}
}




#endif /* BC_INTERNAL_BASE_H_ */
