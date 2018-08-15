/*
 * Shape_Base.h
 *
 *  Created on: Aug 14, 2018
 *      Author: joseph
 */

#ifndef SHAPE_BASE_H_
#define SHAPE_BASE_H_

#include "BlackCat_Internal_Definitions.h"

namespace BC {

template<int dims, class derived>
class Inner_Shape {

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
		  derived& as_derived() 	  { return static_cast<derived&>(*this); }

	auto is() const{
		return as_derived().inner_shape();
	}
public:
	__BCinline__ int size() const { return is()[dims - 1]; }
	__BCinline__ int rows() const { return is()[0]; }
	__BCinline__ int cols() const { return is()[1]; }
	__BCinline__ int dimension(int i) const { return is()[i]; }
	__BCinline__ int outer_dimension() const { return is()[dims - 2]; }
};

template<int dims, class derived>
class Outer_Shape {

	const derived& as_derived() const { return static_cast<const derived&>(*this); }
		  derived& as_derived() 	  { return static_cast<derived&>(*this); }

	auto os()const {
		return as_derived().outer_shape();
	}
public:
	__BCinline__ int leading_dimension(int i) const { return os()[i]; }
};


template<int dimension, class derived>
class Shape_Base
		: public Inner_Shape<dimension, derived>,
		  public Outer_Shape<dimension, derived> {};

}



#endif /* SHAPE_BASE_H_ */
