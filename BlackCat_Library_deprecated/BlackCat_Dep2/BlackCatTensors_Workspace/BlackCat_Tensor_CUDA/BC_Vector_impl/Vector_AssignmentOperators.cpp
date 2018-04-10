#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>& Vector<number_type, TensorOperations>::operator=(Vector<number_type, TensorOperations> && t) {
	this->Tensor<number_type, TensorOperations>::operator=(t);
	return *this;
}
template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>& Vector<number_type, TensorOperations>::operator=(const Vector<number_type, TensorOperations>& t) {
	this->Tensor<number_type, TensorOperations>::operator=(t);
	return *this;
}
template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>& Vector<number_type, TensorOperations>::operator=( const Tensor<number_type, TensorOperations>& t) {
	this->assert_isVector(t);
	this->Tensor<number_type, TensorOperations>::operator=(t);
	return *this;
}
template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>& Vector<number_type, TensorOperations>::operator=(Tensor<number_type, TensorOperations> && t) {
	this->assert_isVector(t);
	this->Tensor<number_type, TensorOperations>::operator=(t);
	return *this;
}

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>& Vector<number_type, TensorOperations>::operator=(const Scalar<number_type, TensorOperations>& s) {
	this->Tensor<number_type, TensorOperations>::operator=(s);
	return * this;
}
