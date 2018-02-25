/*

 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifdef  __CUDACC__
#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "BlackCat_Internal_Definitions.h"
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

	int rank() const 	{ return shadowFailure<int>("int rank() const"); }
	int size() const 	{ return shadowFailure<int>("int size() const"); }
	int rows() const 	{ return shadowFailure<int>("int rows() const"); }
	int cols() const 	{ return shadowFailure<int>("int cols() const"); }
	int LD_rows() const { return shadowFailure<int>("int LD_rows() const"); }
	int LD_cols() const { return shadowFailure<int>("int LD_cols() const");}
	int dimension(int i)		const { return shadowFailure<int>("int dimension(int) const"); }
	void printDimensions() 		const { shadowFailure<>("void printDimensions() const"); }
	void printLDDimensions() 	const { shadowFailure<>("void printLDDimensions() const"); }
	const int* InnerShape() const 			{ return shadowFailure<int*>("auto(const int*) InnerShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
	const int* OuterShape() const 			{ return shadowFailure<int*>("auto(const int*) OuterShape() const  MAY RETURN INT*, _sh<T>, or std::vector<int>, "); }
};

template<class list_type>
struct _sh {
	operator const list_type() const { return dims; }

	_sh(const list_type d) : dims(d) {}
	const list_type dims;
};


}

#endif /* EXPRESSION_BASE_H_ */
#endif
