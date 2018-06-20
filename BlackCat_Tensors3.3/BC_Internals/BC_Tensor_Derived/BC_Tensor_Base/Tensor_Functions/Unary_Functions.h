/*
 * Base_Functions.h
 *
 *  Created on: May 15, 2018
 *      Author: joseph
 */

#ifndef BASE_FUNCTIONS_H_
#define BASE_FUNCTIONS_H_
//This should be included at the bottom of Tensor_Base.h
//Reshape and Chunk are curried functions
//This file defines a set of unary_functions
namespace BC {

//alternate names from transposition
template<class deriv>
static auto transpose(Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.t();
}
//zero and one are the equivalent but delayed evaluation version,
//modifies the calling the tensor, but if and only if evaluated
template<class deriv>
static auto zero(Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::zero());
}
template<class deriv>
static auto one(Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::one());
}

//converts all NaN/Inf values to 0
template<class deriv>
static auto fix(Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::fix());
}
template<class deriv>
static auto abs(const Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::abs());
}
template<class deriv>
static auto negation(const Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::negation());
}

//if 0 return 0 else return 1
template<class deriv>
static auto logical(const Base::Tensor_Operations<deriv>& tensor) {
	 return tensor.un_expr(oper::logical());
}

}


#endif /* BASE_FUNCTIONS_H_ */
