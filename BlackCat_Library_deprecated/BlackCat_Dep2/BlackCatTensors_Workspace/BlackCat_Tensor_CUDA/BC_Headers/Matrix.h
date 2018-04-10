#ifndef BLACKCAT_Matrix_h
#define BLACKCAT_Matrix_h

#include "Tensor.h"

template <typename number_type, class TensorOperations = CPU>
class Matrix : public Tensor<number_type, TensorOperations> {

public:
	Matrix<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& t) 	: Tensor<number_type, TensorOperations>(t) { if(!this->isMatrix()) this->flatten(); };
	Matrix<number_type, TensorOperations>(Tensor<number_type, TensorOperations>&& t) 		: Tensor<number_type, TensorOperations>(t) { if(!this->isMatrix()) this->flatten(); };

	Matrix<number_type, TensorOperations>& operator=(const Tensor<number_type, TensorOperations>& t)			override {this->Tensor<number_type, TensorOperations>::operator=(t); return * this;};
	Matrix<number_type, TensorOperations>& operator=(Tensor<number_type, TensorOperations>&& t) 				override {this->Tensor<number_type, TensorOperations>::operator=(t); return * this;};
	Matrix<number_type, TensorOperations>& operator=(const Scalar<number_type, TensorOperations>& s)			override {this->Tensor<number_type, TensorOperations>::operator=(s); return * this;};
	Matrix<number_type, TensorOperations>& operator=(std::initializer_list<number_type> v) 	override {this->Tensor<number_type, TensorOperations>::operator=(v); return * this;}
	Matrix<number_type, TensorOperations>& operator=(number_type v) 	override {this->Tensor<number_type, TensorOperations>::operator=(v); return * this;}

	Matrix<number_type, TensorOperations>() : Tensor<number_type, TensorOperations>() {/*empty*/}
	explicit Matrix<number_type, TensorOperations>(unsigned m, unsigned n) : Tensor<number_type, TensorOperations>(m, n) {/*empty*/}
	~Matrix<number_type, TensorOperations>() {};

};

#endif
