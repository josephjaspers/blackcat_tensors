/*

 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BlackCat_Internal_Definitions.h"
#include "Expression_Utility_Structs.h"
#include <iostream>

namespace BC {

template<class T, class derived>
struct expression {
private:

	template<class ret = int>
	static ret shadowFailure(std::string method) {
		std::cout << "SHADOW METHOD FAILURE OF REQUIRED METHOD - ENLIGHTENED METHOD: " << method << " -- OVERRIDE THIS METHOD" << std::endl;
		throw std::invalid_argument("MANDATORY METHOD - NOT SHADOWED POTENTIAL INCORRECT CAST BUG");
		return ret();
	}

	const derived& base() const { return static_cast<const derived&>(*this); }
		  derived& base() 		{ return static_cast<	   derived&>(*this); }


public:
//fails with nvcc 9.1 but succeeds with GCC and G++ 6 and 7
//	expression() { static_assert(std::is_trivially_copyable<derived>::value,
//			"EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE"); }


	__BCinline__ static constexpr int DIMS() { return derived::DIMS();    }
	__BCinline__ static constexpr int LAST() { return derived::DIMS() -1; }

	__BCinline__ const auto IS() const { return base().innerShape(); }
	__BCinline__ const auto OS() const { return base().outerShape(); }


	__BCinline__ int dims() const { return DIMS(); }
	__BCinline__ int size() const { return DIMS() > 0 ? OS()[LAST()] : 1;    }
	__BCinline__ int rows() const { return DIMS() > 0 ? IS()[0] : 1; }
	__BCinline__ int cols() const { return DIMS() > 1 ? IS()[1] : 1; }
	__BCinline__ int dimension(int i) const { return DIMS() > i ? base().IS()[i] : 1; }

	__BCinline__ int LD_rows() const { return DIMS() > 0 ? OS()[0] : 1; }
	__BCinline__ int LD_cols() const { return DIMS() > 1 ? OS()[1] : 1; }
	__BCinline__ int LD_dimension(int i) const { return DIMS() > i + 1 ? OS()[i] : 1; }

	__BCinline__ const auto innerShape() const 	{ return shadowFailure<int*>("auto(const int*) innerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	__BCinline__ const auto outerShape() const 	{ return shadowFailure<int*>("auto(const int*) outerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }


	void printDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << innerShape()[i] << "]";
		}
		std::cout << std::endl;
	}
	void printLDDimensions() const {
		for (int i = 0; i < DIMS(); ++i) {
			std::cout << "[" << outerShape()[i] << "]";
		}
		std::cout << std::endl;
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
