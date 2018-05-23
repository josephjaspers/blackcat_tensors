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

namespace BC {

struct BC_Type {};

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

	__BCinline__ static constexpr int LAST() { return derived::DIMS() -1; }

	__BCinline__ const auto IS() const { return base().innerShape(); }
	__BCinline__ const auto OS() const { return base().outerShape(); }

	template<class... integers>
	__BCinline__ const auto operator()(integers... ints) const {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return base()[scal_index(ints...)];
	}

	template<class... integers>
	__BCinline__ auto operator()(integers... ints) {
		static_assert(sizeof...(integers) == DIMS(), "non-definite index given");
		return base()[scal_index(ints...)];
	}

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? OS()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? IS()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? IS()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? IS()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? OS()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? OS()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? OS()[i] : 1; }

	__BCinline__ const auto innerShape() const 	{ return shadowFailure("auto(const int*) innerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	__BCinline__ const auto outerShape() const 	{ return shadowFailure("auto(const int*) outerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }

	void printDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << IS()[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << OS()[i] << "]";
		}
		std::cout << std::endl;
	}
	//---------------------------------------------------UTILITY/IMPLEMENTATION METHODS------------------------------------------------------------//
	template<class... integers> __BCinline__
	int scal_index(integers... ints) const {
		return scal_index(BC::array(ints...)); //fixme should use recursive impl
	}
	template<class... integers> __BCinline__
	int scal_index_reverse(integers... ints) const {
		return  this->scal_index_impl(ints...);
	}

	template<class... integers> __BCinline__
	int scal_index_impl(int front, integers... ints) const {
		return scal_index_impl(ints...) + front * this->LD_dimension(sizeof...(ints) - 1);
	}
	__BCinline__
	int scal_index_impl(int front) const {
		return front;
	}
	template<int D> __BCinline__
	 int scal_index(stack_array<int, D> var) const {
		int index = var[0];
		for(int i = 1; i < var.size(); ++i) {
			index += this->LD_dimension(i - 1) * var[i];
		}
		return index;
	}

	template<int D> __BCinline__
	 int scal_index_reverse(stack_array<int, D> var) const {
		int index = var[D - 1];

		for(int i = 0; i < D - 1; ++i) {
			index += this->LD_dimension(i) * var[D - i - 2];
		}

		return index;
	}


	//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//
	void destroy() {}
	//---------------------------------------------------METHODS THAT NEED TO BE SHADOWED------------------------------------------------------------//
	__BCinline__ auto operator [] (int index) 	  	{ return shadowFailure("operator [] (int index)"); };
	__BCinline__ auto operator [] (int index) const { return shadowFailure("operator [] (int index) const"); };
	__BCinline__ int slice(int i) const { return shadowFailure("const Tensor_Slice(int) const => NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int slice(int i) 	    {  return shadowFailure("Tensor_Slice(int)  =>  NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int row(int i) const 	{ return shadowFailure("auto row(int i) const "); }
	__BCinline__ int row(int i) 	   	{ return shadowFailure("auto row(int i)"); }
	__BCinline__ int col(int i) const 	{ return shadowFailure("auto col(int i) const"); }
	__BCinline__ int col(int i) 	   	{ return shadowFailure("auto col(int i)"); }
};

}

#endif /* EXPRESSION_BASE_H_ */
