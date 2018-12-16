/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef SHAPE_BASE_H_
#define SHAPE_BASE_H_

#include "Internal_Common.h"

namespace BC {
namespace et {

template<class derived>
class Inner_Shape {

     const derived& as_derived() const { return static_cast<const derived&>(*this); }
           derived& as_derived()        { return static_cast<      derived&>(*this); }

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
           derived& as_derived()        { return static_cast<      derived&>(*this); }

     auto bs()const {
        return as_derived().block_shape();
    }
public:
     int size() const { return bs()[derived::DIMS() - 1]; }
     int block_dimension(int i) const { return bs()[i]; }
};

template<class derived>
class Outer_Shape {

     const derived& as_derived() const { return static_cast<const derived&>(*this); }
           derived& as_derived()       { return static_cast<      derived&>(*this); }

     auto os()const {
        return as_derived().outer_shape();
    }
public:
     int leading_dimension(int i) const {
         if (derived::DIMS() == 1)
             if (i == 1)
                 return 1;
         return os()[i];
     }
};



template<class derived>
class Shape_Base
        : public Inner_Shape<derived>,
          public Block_Shape<derived> {};

template<int dimensions, class lamba_inner_shape, class lambda_block_shape>
class Function_Shape
		: lamba_inner_shape,
		  lambda_block_shape,
		  public Inner_Shape<Function_Shape<dimensions, lamba_inner_shape, lambda_block_shape>>,
		  public Block_Shape<Function_Shape<dimensions, lamba_inner_shape, lambda_block_shape>>
  {

	__BCinline__
	static constexpr int DIMS() { return dimensions; }

	__BCinline__
	auto inner_shape() {
		return static_cast<const lamba_inner_shape&>(*this)();
	}
	__BCinline__
	auto block_shape() {
		return static_cast<const lambda_block_shape&>(*this)();
	}
};

}
}

#endif /* SHAPE_BASE_H_ */
