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
#include "../Determiners.h"
#include <iostream>

namespace BC {

template<class T, class derived>
struct expression {
	using type = derived;
	using scalar_type = T;
private:

	template<class ret = int>
	static ret shadowFailure(std::string method) {
		std::cout << "SHADOW METHOD FAILURE OF REQUIRED METHOD - ENLIGHTENED METHOD: " << method << " -- OVERRIDE THIS METHOD" << std::endl;
		throw std::invalid_argument("MANDATORY METHOD - NOT SHADOWED POTENTIAL INCORRECT CAST BUG");
	}
		  derived& asDerived() 		 { return static_cast<	    derived&>(*this); }
	const derived& asDerived() const { return static_cast<const derived&>(*this); }


public:

	__BCinline__ static constexpr int DIMS() { return derived::DIMS(); }
//	expression() { static_assert(std::is_trivially_copyable<derived>::value, "DERIVED VES OF EXPRESSION TYPES MUST BE TRIVIALLY COPYABLE"); }

	__BCinline__ int dims() const 	{ return shadowFailure<int>("int dims() const"); }
	__BCinline__ int size() const 	{ return shadowFailure<int>("int size() const"); }
	__BCinline__ int rows() const 	{ return shadowFailure<int>("int rows() const"); }
	__BCinline__ int cols() const 	{ return shadowFailure<int>("int cols() const"); }
	__BCinline__ int LD_rows() const { return shadowFailure<int>("int LD_rows() const"); }
	__BCinline__ int LD_cols() const { return shadowFailure<int>("int LD_cols() const");}
	__BCinline__ int dimension(int i)		const { return shadowFailure<int>("int dimension(int) const"); }
	 void printDimensions() 		const { shadowFailure<>("void printDimensions() const"); }
	 void printLDDimensions() 	const { shadowFailure<>("void printLDDimensions() const"); }
	__BCinline__ const int* innerShape() const 			{ return shadowFailure<int*>("auto(const int*) innerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	__BCinline__ const int* outerShape() const 			{ return shadowFailure<int*>("auto(const int*) outerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	__BCinline__ int slice(int i) const { return shadowFailure<>("const Tensor_Slice(int) const  => THIS METHOD SHOULD ONLY BE ENABLED FOR TENSOR_CORE"); }
	__BCinline__ int slice(int i) 	   {  return shadowFailure<>("Tensor_Slice(int)  => THIS METHOD SHOULD ONLY BE ENABLED FOR TENSOR_CORE"); }
	__BCinline__ auto operator [] (int index) 	  	{ return shadowFailure<int>("operator [] (int index)"); };
	__BCinline__ auto operator [] (int index) const { return shadowFailure<int>("operator [] (int index) const"); };
	__BCinline__ int row(int i) const 	{ return shadowFailure<>("auto row(int i) const "); }
	__BCinline__ int row(int i) 	   	{ return shadowFailure<>("auto row(int i)"); }
	__BCinline__ int col(int i) const 	{ return shadowFailure<>("auto col(int i) const"); }
	__BCinline__ int col(int i) 	   	{ return shadowFailure<>("auto col(int i)"); }

};

}

#endif /* EXPRESSION_BASE_H_ */
