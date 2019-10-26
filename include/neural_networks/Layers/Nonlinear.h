/*
 * Nonlinear.h
 *
 *  Created on: Jun 8, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_NONLINEAR_H_
#define BLACKCAT_NEURALNETWORK_NONLINEAR_H_

#include "UnaryFunction.h"

namespace BC {
namespace nn {

#define BC_NONLINEAR_DEF(TypeName, funcName)\
auto funcName(BC::size_t inputs) {\
	return Function<BLACKCAT_DEFAULT_SYSTEM_T,\
					typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type,\
					BC::TypeName>(inputs); \
}\
template<class ValueType, class SystemTag>\
auto funcName(SystemTag system, BC::size_t inputs) {\
	return Function<SystemTag, ValueType, BC::TypeName>(inputs); \
}\
template<class SystemTag>\
auto funcName(SystemTag system, BC::size_t inputs) {\
	return Function<SystemTag, typename SystemTag::default_floating_point_type, BC::TypeName>(inputs); \
}

BC_NONLINEAR_DEF(Tanh, tanh)
BC_NONLINEAR_DEF(Logistic, logistic)
BC_NONLINEAR_DEF(Relu, relu)

#undef BC_NONLINEAR_DEF
}
}


#endif /* NONLINEAR_H_ */
