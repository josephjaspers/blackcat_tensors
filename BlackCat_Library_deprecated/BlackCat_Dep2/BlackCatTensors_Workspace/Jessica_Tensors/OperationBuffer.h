#ifndef BlackCat_TensorBones_h
#define BlackCat_TensorBones_h


template<typename T>
class Tensor_Bones {

	T* data;

public:
	virtual int size() const = 0;
	virtual int order() const = 0;
	virtual int order(int index) const = 0;

	virtual void zero() = 0;
	virtual void fill(T value) = 0;
	virtual void print() const = 0;
	virtual void printDimension() const = 0;

	virtual T min() const = 0;
	virtual T max() const = 0;
	virtual std::pair<T, int> max_index() const = 0;
	virtual std::pair<T, int> min_index() const = 0;

	virtual ~Tensor_Bones() = 0;
};

#include "PointwiseOperations.h"

template<typename T, typename oper>
class bTensor : public Tensor_Bones<T> {


	int size() const override;
	int order() const override;
	int order(int index) const override;

	bTensor<T, oper> operator * (const Tensor_Bones<T>& tens) const;
	bTensor<T, oper> operator & (const Tensor_Bones<T>& tens) const;
	bTensor<T, oper> operator / (const Tensor_Bones<T>& tens) const;
	bTensor<T, oper> operator + (const Tensor_Bones<T>& tens) const;
	bTensor<T, oper> operator - (const Tensor_Bones<T>& tens) const;
};


#endif
