/*
 * BC_Tensor_BaseClass_Tensor.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_BASECLASS_TENSOR_H_
#define BC_TENSOR_BASECLASS_TENSOR_H_


#include "BC_Tensor_BaseClass_Lv1_Vector.h"

template<class T, class ml = CPU, int... dimensions>
class Tensor : public Vector<T, ml, dimensions...> {

};


#endif /* BC_TENSOR_BASECLASS_TENSOR_H_ */
