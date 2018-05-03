/*

 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "BC_Utility/Internal_Shapes.h"
#include "BC_Utility/Utility.h"

#include <iostream>

namespace BC {


struct BC_Type {};

template<class derived>
struct expression_base : BC_Type {
private:

	template<class ret = int>
	static ret shadowFailure(std::string method) {
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
	__BCinline__ static constexpr int PARENT_DIMS() { return derived::DIMS();    }

	__BCinline__ static constexpr int LAST() { return derived::DIMS() -1; }
	__BCinline__ const auto IS() const { return base().innerShape(); }
	__BCinline__ const auto OS() const { return base().outerShape(); }

	template<class... integers>
	__BCinline__ const auto operator()(integers... ints) const { return base()[scal_index(ints...)]; }
	template<class... integers>
	__BCinline__ auto operator()(integers... ints) { return base()[scal_index(ints...)]; }

	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? OS()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? IS()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? IS()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? IS()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? OS()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? OS()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? OS()[i] : 1; }

	__BCinline__ const auto innerShape() const 	{ return shadowFailure<int*>("auto(const int*) innerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	__BCinline__ const auto outerShape() const 	{ return shadowFailure<int*>("auto(const int*) outerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }

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

	//base case
	template<int dim = 0> __BCinline__
	int scal_index(int curr) const {
		if (dim == 0)
			return curr;
		else
			return curr * this->dimension(dim - 1);
	}

	template<int dim = 0, class ... integers> __BCinline__
	int scal_index(int curr, integers ... ints) const {
		static constexpr bool int_sequence = MTF::is_integer_sequence<integers...>;
		static_assert(int_sequence, "MUST BE INTEGER LIST");
		if (dim == 0)
			return curr + scal_index<dim + 1>(ints...);
		else
			return curr * this->dimension(dim - 1) + scal_index<dim + 1>(ints...);
	}

	template<class... integers>
	int point_index(integers... ints) const {
		auto var = array(ints...);
		int index = this->rows();
		for (int i = 1; i < var.size(); ++i) {
			index += var[i] * this->dimension(i);
		}
		return index;
	}

	//---------------------------------------------------METHODS THAT MAY NEED TO BE SHADOWED------------------------------------------------------------//
	void destroy() {}
	//---------------------------------------------------METHODS THAT NEED TO BE SHADOWED------------------------------------------------------------//
	__BCinline__ auto operator [] (int index) 	  	{ return shadowFailure<int>("operator [] (int index)"); };
	__BCinline__ auto operator [] (int index) const { return shadowFailure<int>("operator [] (int index) const"); };
	__BCinline__ int slice(int i) const { return shadowFailure<>("const Tensor_Slice(int) const => NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int slice(int i) 	    {  return shadowFailure<>("Tensor_Slice(int)  =>  NOT ENABLED FOR ALL EXPRESSIONS"); }
	__BCinline__ int row(int i) const 	{ return shadowFailure<>("auto row(int i) const "); }
	__BCinline__ int row(int i) 	   	{ return shadowFailure<>("auto row(int i)"); }
	__BCinline__ int col(int i) const 	{ return shadowFailure<>("auto col(int i) const"); }
	__BCinline__ int col(int i) 	   	{ return shadowFailure<>("auto col(int i)"); }
	template<class... integers>
	__BCinline__ auto reshape(integers...) { return shadowFailure<>("auto reshape"); }

};

}

#endif /* EXPRESSION_BASE_H_ */
//
//template<class rv> __BCinline__ auto operator + (const pbase<rv>& tensor) const { return binary_expression<lv, rv, add>(base(), tensor.base()); }
//template<class rv> __BCinline__ auto operator - (const pbase<rv>& tensor) const { return binary_expression<lv, rv, sub>(base(), tensor.base()); }
//template<class rv> __BCinline__ auto operator / (const pbase<rv>& tensor) const { return binary_expression<lv, rv, div>(base(), tensor.base()); }
//template<class rv> __BCinline__ auto operator * (const pbase<rv>& tensor) const { return binary_expression<lv, rv, mul>(base(), tensor.base()); }
////asterix is multiplication in this context
