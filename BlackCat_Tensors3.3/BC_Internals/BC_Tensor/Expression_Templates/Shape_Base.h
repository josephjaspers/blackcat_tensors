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
namespace internal {

template<class derived>
class Inner_Shape {

	 const derived& as_derived() const { return static_cast<const derived&>(*this); }
		   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

	 auto is() const{
		return as_derived().inner_shape();
	}
public:
	 int rows() const { return is()[0]; }
	 int cols() const { return is()[1]; }
	 int dimension(int i) const { return is()[i]; }
	 int outer_dimension() const { return is()[derived::DIMS() - 2]; }
};

template<class derived>
class Block_Shape {

	 const derived& as_derived() const { return static_cast<const derived&>(*this); }
		   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }

	 auto bs()const {
		return as_derived().block_shape();
	}
public:
	 int size() const { return bs()[derived::DIMS() - 1]; }
	 int block_dimension(int i) const { return bs()[i]; }
};


template<class derived>
class Shape_Base
		: public Inner_Shape<derived>,
		  public Block_Shape<derived> {};

}
}


#endif /* SHAPE_BASE_H_ */
