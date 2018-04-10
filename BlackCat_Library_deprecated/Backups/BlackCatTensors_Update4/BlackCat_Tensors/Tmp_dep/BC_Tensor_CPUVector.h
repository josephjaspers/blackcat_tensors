///*
// * BC_Tensor_CPUVector.h
// *
// *  Created on: Dec 4, 2017
// *      Author: joseph
// */
//
//#ifndef BC_TENSOR_CPUVECTOR_H_
//#define BC_TENSOR_CPUVECTOR_H_
//
//#include "BC_Tensor_InheritLv5_Core.h"
//
//template<class T, int row >
//class CPU_Vector : Tensor_Core<T, CPU, row> {
//
//	CPU_Vector<T, row>& operator = (const Tensor_King<T, CPU, row>& tens) {
//			return * this;
//		}
//
//		template<class U>
//		CPU_Vector<T, row>& operator = (const Tensor_King<U, CPU, row>& tens) {
//			CPU::copy(this->data(), tens.data(), this->size());
//			return * this;
//		}
//
//};
//
//
//
//#endif /* BC_TENSOR_CPUVECTOR_H_ */
