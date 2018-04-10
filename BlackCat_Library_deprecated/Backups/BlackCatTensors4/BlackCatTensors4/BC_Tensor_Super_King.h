/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 20, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_KING_H_
#define BC_TENSOR_SUPER_KING_H_

#include "BC_Tensor_Super_Shape.h"
#include "BC_Tensor_Super_Ace.h"
#include "BC_Expression_Functors.h"
#include "BlackCat_MetaTemplateFunctions.h"
#include "BC_Math_CPU.h"
#include <type_traits>
/*
 * Defines the internal data type based upon template specializations.
 * Generally either a numerical type or some type of expression function.
 */
template<class oper, class ml, class l, class r, int ... dimensions>
class binary_expression;


template<class T, template<class, class, class, class, int...> class binExpr = binary_expression, int... dimensions>
struct Tensor_King : public Shape<dimensions...>, public Tensor_Ace<T> {

	using functor_type = typename Tensor_Ace<T>::functor_type;

};

#endif /* BC_TENSOR_SUPER_KING_H_ */
