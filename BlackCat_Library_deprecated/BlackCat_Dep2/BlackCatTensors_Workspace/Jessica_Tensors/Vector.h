/*
 * Vector.h
 *
 *  Created on: Oct 11, 2017
 *      Author: joseph
 */

#ifndef VECTOR_H_
#define VECTOR_H_


template<typename T>
class Vector : Tensor_Bones<T>  {
protected:
	//T* tensor;
public:


	template<typename oper>
	Vector& operator = (const bTensor<T, oper>);
	Vector& operator = (const Tensor_Bones<T> tensor);

	bTensor<T, dot> operator * (const Tensor_Bones<T>& tens) const;
	bTensor<T, mul> operator & (const Tensor_Bones<T>& tens) const;
	bTensor<T, div> operator / (const Tensor_Bones<T>& tens) const;
	bTensor<T, add> operator + (const Tensor_Bones<T>& tens) const;
	bTensor<T, sub> operator - (const Tensor_Bones<T>& tens) const;

;
};

#endif /* VECTOR_H_ */
