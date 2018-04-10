/*
 * BC_Internal_Include.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_INTERNALS_INCLUDE_H_
#define BC_INTERNALS_INCLUDE_H_

#include "BC_Expression_Binary_Functors.h"
#include "BC_Expression_Binary_Pointwise_Scalar_L.h"
#include "BC_Expression_Binary_Pointwise_Scalar_R.h"

#include "BC_Expression_Binary_Pointwise_Same.h"
#include "BC_Mathematics_CPU.h"
#include "BC_MetaTemplate_UtilityMethods.h"

template<class T, class ml>
class Scalar;

template<class T, class ml, int... dimensions>
class Vector;

template<class T, class ml, int... dimensions>
class Matrix;

template<class T, class ml, int... dimensions>
class Cube;

template<class T, class ml, int... dimensions>
class Tensor;

template<class T, class ml, int... dimensions>
class Tensor_Super;

template<class T>
struct expression;

template<class T, class oper>
struct unary_expression;

template<class T, class oper, class lv, class rv>
struct binary_expression;


namespace printHelper {

	template<int curr, int ... stack>
	struct f {
		void fill(int* ary) {
			ary[0] = curr;
			f<stack...>().fill(&ary[1]);
		}
	};
	template<int dim>
	struct f<dim> {
		void fill(int* ary) {
			ary[0] = dim;
		}
	};
}

#endif /* BC_INTERNALS_INCLUDE_H_ */
