/*
 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include <iostream>
#include <type_traits>
#include <cmath>

namespace BC {
namespace internal {

template<class derived>
struct expression_base : BC_Type {
private:

	static int shadowFailure(std::string method) {
		std::cout << "SHADOW METHOD FAILURE OF REQUIRED METHOD - ENLIGHTENED METHOD: " << method << " -- OVERRIDE THIS METHOD" << std::endl;
		throw std::invalid_argument("MANDATORY METHOD - NOT SHADOWED POTENTIAL INCORRECT CAST BUG");
	}

	__BCinline__ const derived& base() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& base() 		 { return static_cast<	    derived&>(*this); }

public:
	operator 	   auto&()       { return base(); }
	operator const auto&() const { return base(); }
//	Expression_Core_Base() { static_assert(std::is_trivially_copyable<derived>::value, "EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE"); }

	__BCinline__ static constexpr int DIMS() { return derived::DIMS();    }
	__BCinline__ static constexpr bool ASSIGNABLE() { return false; }
	__BCinline__ static constexpr bool ITERATOR() { return -1; } 		//this will cause compile_failure if not shadowed (mandatory to shadow)
	__BCinline__ static constexpr bool INJECTABLE() { return false; }	//Injectable is boolean value determining


	__BCinline__ static constexpr int last() { return derived::DIMS() -1; }

	__BCinline__ const auto IS() const { return base().inner_shape(); }
	__BCinline__ const auto OS() const { return base().outer_shape(); }

	template<class... integers>
	__BCinline__ const auto operator()(integers... ints) const {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return base()[dims_to_index(ints...)];
	}

	template<class... integers>
	__BCinline__ auto operator()(integers... ints) {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return base()[dims_to_index(ints...)];
	}

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? OS()[last()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? IS()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? IS()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? IS()[i] : 1; }
	__BCinline__ int outer_dimension() const { return dimension(DIMS() - 1); }
	__BCinline__ auto iterator_limit(int i) { return this->dimension(i); }
	__BCinline__ auto iterator_increment(int i) { return 1; }

	__BCinline__ int ld1() const { return DIMS() > 0 ? OS()[0] : 1; }
	__BCinline__ int ld2() const { return DIMS() > 1 ? OS()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? OS()[i] : 1; }

	__BCinline__ const auto inner_shape() const 	{ return shadowFailure("inner_shape not shadowed"); }
	__BCinline__ const auto outer_shape() const 	{ return shadowFailure("outer_shape not shadowed"); }

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
		return dims_to_index_impl(ints...) + front * this->LD_dimension(sizeof...(ints) - 1);
	}
	__BCinline__ int dims_to_index_impl(int front) const {
		return front;
	}
	template<int D> __BCinline__ int dims_to_index(stack_array<int, D> var) const {
		int index = var[0];
		for(int i = 1; i < var.size(); ++i) {
			index += this->LD_dimension(i - 1) * var[i];
		}
		return index;
	}

	template<int D> __BCinline__ int dims_to_index_reverse(stack_array<int, D> var) const {
		int index = var[D - 1];

		for(int i = 0; i < D - 2; ++i) {
			index += this->LD_dimension(i) * var[D - i - 2];
		}

		return index;
	}

	__BCinline__ auto index_to_dims(int index) const {

		stack_array<int, DIMS()> dim_set;
		for (int i = DIMS() - 2; i >= 0; --i) {
			dim_set[i + 1] = index / LD_dimension(i);
			index -= (int)(index / LD_dimension(i)) * LD_dimension(i);
		}
		dim_set[0] = index;

		return dim_set;
	}


	//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//
	void destroy() {}
	//---------------------------------------------------METHODS THAT NEED TO BE SHADOWED------------------------------------------------------------//
	__BCinline__ auto operator [] (int index) 	  	{ return shadowFailure("operator [] (int index)"); };
	__BCinline__ auto operator [] (int index) const { return shadowFailure("operator [] (int index) const"); };
	__BCinline__ int slice(int i) const { return shadowFailure("const Tensor_Slice(int) const => NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int slice(int i) 	    { return shadowFailure("Tensor_Slice(int)  =>  NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int row(int i) const 	{ return shadowFailure("auto row(int i) const "); }
	__BCinline__ int row(int i) 	   	{ return shadowFailure("auto row(int i)"); }
	__BCinline__ int col(int i) const 	{ return shadowFailure("auto col(int i) const"); }
	__BCinline__ int col(int i) 	   	{ return shadowFailure("auto col(int i)"); }
};
}
}

#endif /* EXPRESSION_BASE_H_ */
