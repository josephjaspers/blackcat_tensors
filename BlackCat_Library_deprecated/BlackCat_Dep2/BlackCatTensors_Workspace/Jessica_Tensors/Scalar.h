#ifndef BlackCat_Scalar_h
#define BlackCat_Scalar_h

#include "Tensor_Super.h"
#include "OperationBuffer.h"

template <typename T>
class Scalar : Tensor_Bones<T> {

public:
	//T* tensor;

	int size() const { return 1; };
	int order() const { return 1; };
	int order(int index) const { return 1; };

	int zero(); //memset 0
	int fill(T value); //memset value
	int print() const; //cout //port copy if needed
	int printDimensions() const; //print

	T max() const;
	T min() const;

	std::pair<T, int> max_index() const;
	std::pair<T, int> min_index() const;

	bTensor<T, mul_scal> operator * (const Tensor_Bones<T>& tens) const;
	bTensor<T, mul_scal> operator & (const Tensor_Bones<T>& tens) const;
	bTensor<T, div_scal> operator / (const Tensor_Bones<T>& tens) const;
	bTensor<T, add_scal> operator + (const Tensor_Bones<T>& tens) const;
	bTensor<T, sub_scal> operator - (const Tensor_Bones<T>& tens) const;

	T operator + (const Scalar& s) const;
	T operator - (const Scalar& s) const;
	T operator / (const Scalar& s) const;
	T operator & (const Scalar& s) const;
	T operator * (const Scalar& s) const;
};

#endif
