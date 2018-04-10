/*
 * BC_Tensor_Primary_Tensor.h
 *
 *  Created on: Nov 30, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_PRIMARY_TENSOR_H_
#define BC_TENSOR_PRIMARY_TENSOR_H_

#include "BC_Tensor_Super_Jack.h"

template<class T, class ml, int... dimensions>
class Tensor : public Tensor_Jack<T, ml, dimensions...> {

};


#endif /* BC_TENSOR_PRIMARY_TENSOR_H_ */
