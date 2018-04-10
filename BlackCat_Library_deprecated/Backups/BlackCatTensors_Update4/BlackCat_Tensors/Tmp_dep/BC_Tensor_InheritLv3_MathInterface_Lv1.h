/*
 * BC_Tensor_InheritLv2_Operand_Interface.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_INHERITLV3_MATHINTERFACE_LV1_H_
#define BC_TENSOR_INHERITLV3_MATHINTERFACE_LV1_H_

#include "../BC_Tensor_InheritLv2_Ace.h"

template<class T, class ml, int... dimensions>
struct Tensor_Upper_MathInterface : Tensor_Ace<T, ml, dimensions...>{

	/*
	 * Class designed for Specializations for Mathematical operations that are
	 * dependent upon the dimensionality of the Tensors (IE dotproduct)
	 */

	//Note* All Scalar by Scalar operations will NOT use lazy evaluation
	//Should use immediate evaluation
};


#endif /* BC_TENSOR_INHERITLV3_MATHINTERFACE_LV1_H_ */
