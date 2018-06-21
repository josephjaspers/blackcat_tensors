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

#include <iostream>
#include <type_traits>

namespace BC {
namespace internal {

template<class derived>
class expression_base : BC_Type {

	__BCinline__ static constexpr int  DIMS()       { return derived::DIMS(); }
	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

	__BCinline__ const auto IS() const { return as_derived().inner_shape(); }
	__BCinline__ const auto OS() const { return as_derived().outer_shape(); }

public:

	operator 	   auto&()       { return as_derived(); }
	operator const auto&() const { return as_derived(); }

	expression_base() { static_assert(std::is_trivially_copyable<derived>::value, "EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE"); }


	template<class... integers>
	__BCinline__ const auto operator()(integers... ints) const {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return as_derived()[dims_to_index(ints...)];
	}

	template<class... integers>
	__BCinline__ auto operator()(integers... ints) {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return as_derived()[dims_to_index(ints...)];
	}

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? OS()[derived::DIMS() -1] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? IS()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? IS()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? IS()[i] : 1; }
	__BCinline__ int outer_dimension() const { return dimension(DIMS() - 1); }
	__BCinline__ auto iterator_limit(int i) { return this->dimension(i); }
	__BCinline__ auto iterator_increment(int i) { return 1; }

	__BCinline__ int ld1() const { return DIMS() > 0 ? OS()[0] : 1; }
	__BCinline__ int ld2() const { return DIMS() > 1 ? OS()[1] : 1; }
	__BCinline__ int leading_dimension(int i) const { return DIMS() > i + 1 ? OS()[i] : 1; }

	void print_dimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << IS()[i] << "]";
		}
		std::cout << std::endl;
	}
	void print_leading_dimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << OS()[i] << "]";
		}
		std::cout << std::endl;
	}


	//---------------------------------------------------UTILITY/IMPLEMENTATION METHODS------------------------------------------------------------//
	template<class... integers> __BCinline__
	int dims_to_index(integers... ints) const {
		return dims_to_index(BC::array(ints...)); //fixme should use recursive impl
	}
	template<class... integers> __BCinline__
	int dims_to_index_reverse(integers... ints) const {
		return  this->dims_to_index_impl(ints...);
	}

	template<class... integers> __BCinline__
	int dims_to_index_impl(int front, integers... ints) const {
		return dims_to_index_impl(ints...) + front * this->leading_dimension(sizeof...(ints) - 1);
	}
	__BCinline__ int dims_to_index_impl(int front) const {
		return front;
	}
	template<int D> __BCinline__ int dims_to_index(stack_array<int, D> var) const {
		int index = var[0];
		for(int i = 1; i < var.size(); ++i) {
			index += this->leading_dimension(i - 1) * var[i];
		}
		return index;
	}

	template<int D> __BCinline__ int dims_to_index_reverse(stack_array<int, D> var) const {
		int index = var[D - 1];
		for(int i = 0; i < D - 2; ++i) {
			index += this->leading_dimension(i) * var[D - i - 2];
		}
		return index;
	}
	//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//
	void destroy() const {}
	//---------------------------------------------------METHODS THAT NEED TO BE SHADOWED------------------------------------------------------------//
	__BCinline__ _scalar<derived> operator [] (int index) 	  	{ throw std::invalid_argument("method: \" operator [] (int index) \" not shadowed"); return 0;};
	__BCinline__ _scalar<derived> operator [] (int index) const { throw std::invalid_argument("method: \" operator [] (int index) const\" not shadowed"); return 0;};
	//-------------------------------------------------tree re-ordering methods---------------------------------------------------------//
};
}
}

#endif /* EXPRESSION_BASE_H_ */
