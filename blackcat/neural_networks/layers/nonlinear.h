/*
 * Nonlinear.h
 *
 *  Created on: Jun 8, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_NONLINEAR_H_
#define BLACKCAT_NEURALNETWORK_NONLINEAR_H_

#include "unaryfunction.h"

namespace bc {
namespace nn {

#define BC_NONLINEAR_DEF(TypeName, funcName)\
auto funcName(bc::size_t inputs) {\
	return Function<BLACKCAT_DEFAULT_SYSTEM_T,\
					typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type,\
					bc::TypeName>(bc::Dim<1>{inputs}); \
}\
template<class ValueType, class SystemTag>\
auto funcName(SystemTag system, bc::size_t inputs) {\
	return Function<SystemTag, ValueType, bc::TypeName>(bc::Dim<1>{inputs}); \
}\
template<class SystemTag>\
auto funcName(SystemTag system, bc::size_t inputs) {\
	return Function<SystemTag, typename SystemTag::default_floating_point_type, bc::TypeName>(bc::Dim<1>{inputs}); \
}\
template<class ValueType, class SystemTag, int X>\
auto funcName(SystemTag system, bc::Dim<X> inputs) {\
	return Function<                                           \
			SystemTag,                                          \
			ValueType,                                           \
			bc::TypeName,                                           \
			bc::traits::Integer<X>>(inputs);                         \
}                                           \
template<class SystemTag, int X>                                         \
auto funcName(SystemTag system, bc::Dim<X> inputs) {                 \
	return Function<                                           \
			SystemTag,                                           \
			typename SystemTag::default_floating_point_type,      \
			bc::TypeName,                                           \
			bc::traits::Integer<X>>(inputs);                        \
}

BC_NONLINEAR_DEF(Tanh, tanh)
BC_NONLINEAR_DEF(Logistic, logistic)
BC_NONLINEAR_DEF(Relu, relu)
BC_NONLINEAR_DEF(SoftPlus, softplus)
BC_NONLINEAR_DEF(Mish, mish)

#undef BC_NONLINEAR_DEF
}
}


#endif /* NONLINEAR_H_ */
