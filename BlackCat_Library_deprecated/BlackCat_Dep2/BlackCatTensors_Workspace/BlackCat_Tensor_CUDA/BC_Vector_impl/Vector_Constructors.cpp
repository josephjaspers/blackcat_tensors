#include "Vector.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>::Vector(const std::initializer_list<number_type>& vector) : Matrix<number_type, TensorOperations>(vector.size(), 1) {
	CPU::copy(this->data(), vector.begin(), vector.size());
}

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>::Vector(const Vector<number_type, TensorOperations>& vec) : Matrix<number_type, TensorOperations>(vec.size(), 1) {
	CPU::copy(this->data(), vec.data(), this->size());
}

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>::Vector(Vector<number_type, TensorOperations>&& t) : Matrix<number_type, TensorOperations>(std::move(t)) {
	this->flatten();
}

template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>::Vector(const Tensor<number_type, TensorOperations>& t) : Matrix<number_type, TensorOperations>(t) {
	//this->assert_isVector(t);
	this->flatten();

}
template<typename number_type, class TensorOperations>
Vector<number_type, TensorOperations>::Vector(Tensor<number_type, TensorOperations>&& t) : Matrix<number_type, TensorOperations>(std::move(t)) {
//	this->assert_isVector(t);
	this->flatten();
}
