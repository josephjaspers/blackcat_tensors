/*

 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifdef  __CUDACC__
#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_
namespace BC {

#include <cuda.h>
#include "BlackCat_Internal_Definitions.h"


template<class T, class derived>
struct expression {
	using type = derived;
	using scalar_type = T;

		  derived& asDerived() 		 { return static_cast<	    derived&>(*this); }
	const derived& asDerived() const { return static_cast<const derived&>(*this); }


};

template<class T>
auto addressOf(const T& param, int offset) {
	return param.addressOf(offset);
}
//	auto operator [] (int index) -> decltype(asDerived()[index]) { return asDerived()[index]; }
//	auto operator [] (int index) const -> decltype(asDerived()[index]) { return asDerived()[index]; }

//	int rank() const { return asDerived().rank(); }
//	int size() const { return asDerived().size(); }
//	int rows() const { return asDerived().rows(); }
//	int cols() const { return asDerived().cols(); }
//	int LD_rows() const { return asDerived().LD_rows(); }
//	int LD_cols() const { return asDerived().LD_cols(); }
//	int dimension(int i)		const { return asDerived().dimension(i);    }
//	int LD_dimension(int i) 	const { return asDerived().LD_dimension(i); }
//	void printDimensions() 		const { asDerived().printDimensions();   }
//	void printLDDimensions()	const { asDerived().printLDDimensions(); }
//
//	auto InnerShape()  const { return asDerived().InnerShape(); }
//	auto OuterShape()  const { return asDerived().OuterShape(); }
}

#endif /* EXPRESSION_BASE_H_ */
#endif
