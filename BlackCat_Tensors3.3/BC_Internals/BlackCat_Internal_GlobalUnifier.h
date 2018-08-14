/*
 * BC_Internal_Include.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BLACKCAT_INTERNAL_GLOBALUNIFIER_H_
#define BLACKCAT_INTERNAL_GLOBALUNIFIER_H_

#define BC_ARRAY_ONLY(literal) static_assert(BC::is_array<functor_of<derived>>(), "BC Method: '" literal "' is only supported by Array_Base classes")


#include "BC_Tensor/Tensor.h"
#include "BC_MathLibraries/CPU.h"
#include "BC_MathLibraries/GPU.cu"


#endif /* BLACKCAT_INTERNAL_GLOBALUNIFIER_H_ */
