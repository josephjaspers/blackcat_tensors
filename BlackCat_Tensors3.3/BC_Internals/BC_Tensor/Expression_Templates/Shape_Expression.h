/*
 * Shape_Interface.h
 *
 *  Created on: Sep 3, 2018
 *      Author: joseph
 */

#ifndef SHAPE_INTERFACE_H_
#define SHAPE_INTERFACE_H_

#include "Shape_Base.h"

namespace BC {

template<class derived>
struct Shape_Expression : Shape_Base<derived>{

	Shape_Expression() {

		static_assert(!std::is_same<decltype(std::declval<derived>().inner_shape()), void>::value, "EXPRESSION_MUST DEFINE inner_shape()");
		static_assert(!std::is_same<decltype(std::declval<derived>().block_shape()), void>::value, "EXPRESSION_MUST DEFINE block_shape()");

	}


};

}



#endif /* SHAPE_INTERFACE_H_ */
