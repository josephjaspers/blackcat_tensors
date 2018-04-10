/*
 * BC_Tensor_BaseClass_Lv0_Scalar.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_BASECLASS_LV0_SCALAR_H_
#define BC_TENSOR_BASECLASS_LV0_SCALAR_H_

#include "BC_Tensor_InheritLv4_Queen.h"
#include "BC_Tensor_InheritLv5_Core.h"
template<class T, class ml>
class Scalar : public Tensor_Core<T, ml, 1> {

	template<class, class, int...>
	friend class Vector;

public:

	Scalar<T, ml>(T* scalar) : Tensor_Queen<T, ml, 1>(scalar) {}


	Scalar<T, ml>& operator = (T value) { ml::set(this->data(), value); return * this;}

	template<typename U>
	Scalar<T, ml>& operator = (const Scalar<U, ml>& scalar) { ml::set(this->data(), scalar.data()); return *this; }
};

#endif /* BC_TENSOR_BASECLASS_LV0_SCALAR_H_ */
