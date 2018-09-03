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

template<class derived>
class Inner_Shape {

	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

	__BCinline__ auto is() const{
		return as_derived().inner_shape();
	}
public:
	__BCinline__ int rows() const { return is()[0]; }
	__BCinline__ int cols() const { return is()[1]; }
	__BCinline__ int dimension(int i) const { return is()[i]; }
	__BCinline__ int outer_dimension() const { return is()[derived::DIMS() - 2]; }
};

template<class derived>
class Block_Shape {

	__BCinline__ const derived& as_derived() const { return static_cast<const derived&>(*this); }
	__BCinline__	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

	__BCinline__ auto bs()const {
		return as_derived().block_shape();
	}
public:
	__BCinline__ int size() const { return bs()[derived::DIMS() - 1]; }
	__BCinline__ int block_dimension(int i) const { return bs()[i]; }
};


template<class derived>
class Shape_Base
		: public Inner_Shape<derived>,
		  public Block_Shape<derived> {};

}



#endif /* SHAPE_BASE_H_ */
