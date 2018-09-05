/*
 * Module_Functions.h
 *
 *  Created on: May 15, 2018
 *      Author: joseph
 */

#ifndef BASE_FUNCTIONS_H_
#define BASE_FUNCTIONS_H_
//This should be included at the bottom of Tensor_Module.h
//Reshape and Chunk are curried functions
//This file defines a set of unary_functions
namespace BC {

//alternate names from transposition
template<class deriv>
static auto transpose(Module::Tensor_Operations<deriv>& tensor) {
	return tensor.t();
}
//zero and one are the equivalent but delayed evaluation version,
//modifies the calling the tensor, but if and only if evaluated
template<class deriv>
static auto zero(Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::zero());
}
template<class deriv>
static auto one(Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::one());
}

//converts all NaN/Inf values to 0
template<class deriv>
static auto fix(Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::fix());
}
template<class deriv>
static auto abs(const Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::absolute());
}
template<class deriv>
static auto negation(const Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::negation());
}

//if 0 return 0 else return 1
template<class deriv>
static auto logical(const Module::Tensor_Operations<deriv>& tensor) {
	return tensor.un_expr(Module::oper::logical());
}

template<class deriv>
static auto normalize(const Module::Tensor_Operations<deriv>& tensor, scalar_of<deriv> min, scalar_of<deriv> max) {
	return tensor.un_expr(Module::oper::norm<scalar_of<deriv>>(min, max));
}

}


#endif /* BASE_FUNCTIONS_H_ */
