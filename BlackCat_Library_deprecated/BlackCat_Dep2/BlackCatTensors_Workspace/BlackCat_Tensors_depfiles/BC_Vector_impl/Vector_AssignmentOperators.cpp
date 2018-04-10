#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"



template<typename number_type>
Vector<number_type>& Vector<number_type>::operator=(Vector<number_type> && t) {
	this->Tensor<number_type>::operator=(t);
	return *this;
}
template<typename number_type>
Vector<number_type>& Vector<number_type>::operator=(const Vector<number_type>& t) {
	this->Tensor<number_type>::operator=(t);
	return *this;
}
template<typename number_type>
Vector<number_type>& Vector<number_type>::operator=( const Tensor<number_type>& t) {
	this->assert_isVector(t);
	this->Tensor<number_type>::operator=(t);
	return *this;
}
template<typename number_type>
Vector<number_type>& Vector<number_type>::operator=(Tensor<number_type> && t) {
	this->assert_isVector(t);
	this->Tensor<number_type>::operator=(t);
	return *this;
}

template<typename number_type>
Vector<number_type>& Vector<number_type>::operator=(const Scalar<number_type>& s) {
	this->Tensor<number_type>::operator=(s);
	return * this;

}
