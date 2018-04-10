/*
 * Shape.h
 *
 *  Created on: Dec 28, 2017
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_
#include "Static_Shape_Outer_Impl.h"
#include "../BlackCat_Internal_GlobalUnifier.h"

namespace BC{

	template<class Static_Inner_Shape, class Static_Outer_Shape = typename DEFAULT_LD<Static_Inner_Shape>::type>
	struct Shape : Static_Inner_Shape, Static_Outer_Shape {

		static constexpr bool isContinuous = BC_MTF::is_same<Static_Outer_Shape, typename DEFAULT_LD<Static_Inner_Shape>::type>::conditional;
		static constexpr int  RANK = Static_Inner_Shape::RANK;
		static constexpr int  SIZE = Static_Inner_Shape::size();

		using outer_shape = Static_Outer_Shape;
		using inner_shape = Static_Inner_Shape;
	};

}

#endif /* SHAPE_H_ */
