#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Vector<number_type>::Vector(const std::initializer_list<number_type>& vector) : Matrix<number_type>(vector.size(), 1) {
	Tensor_Operations<number_type>::copy(this->data(), vector.begin(), vector.size());
}

template<typename number_type>
Vector<number_type>::Vector(const Vector<number_type>& vec) : Matrix<number_type>(vec.size(), 1) {
	Tensor_Operations<number_type>::copy(this->data(), vec.data(), this->size());
}

template<typename number_type>
Vector<number_type>::Vector(Vector<number_type>&& t) : Matrix<number_type>(std::move(t)) {
}

template<typename number_type>
Vector<number_type>::Vector(const Tensor<number_type>& t) : Matrix<number_type>(t) {
	//this->assert_isVector(t);
	this->flatten();

}
template<typename number_type>
Vector<number_type>::Vector(Tensor<number_type>&& t) : Matrix<number_type>(std::move(t)) {
//	this->assert_isVector(t);
	this->flatten();
}
