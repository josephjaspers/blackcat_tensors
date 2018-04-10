#include "Scalar.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
template<typename number_type, typename TensorOperations>


Scalar<number_type, TensorOperations>::Scalar(const Scalar<number_type, TensorOperations>& s) {
	CPU::initialize(scalar, 1);
	CPU::fill(scalar, s.scalar[0], 1);
}

template<typename number_type, typename TensorOperations>
Scalar<number_type, TensorOperations>::Scalar(Scalar<number_type, TensorOperations> && s) {
	if (s.ownership && ownership)
	scalar = s.scalar;
	else {
		CPU::fill(scalar, s.scalar[0], 1);
	}
}

template<typename number_type, typename TensorOperations>
Scalar<number_type, TensorOperations>::Scalar(number_type value) {
	CPU::initialize(scalar, 1);
	CPU::fill(scalar, value, 1);
}

template<typename number_type, typename TensorOperations>
Scalar<number_type, TensorOperations>& Scalar<number_type, TensorOperations>::operator =(const Scalar<number_type, TensorOperations>& s) {
	CPU::fill(scalar, s.scalar[0], 1);
}

template<typename number_type, typename TensorOperations>
Scalar<number_type, TensorOperations>& Scalar<number_type, TensorOperations>::operator =(Scalar<number_type, TensorOperations> && s) {

	if (s.ownership && ownership) {
		CPU::destruction(this->scalar);
		this->scalar = s.scalar;
	} else {
		CPU::fill(scalar, s.scalar[0], 1);
	}
}

template<typename number_type, typename TensorOperations>
Scalar<number_type, TensorOperations>& Scalar<number_type, TensorOperations>::operator =(number_type s) {
	CPU::fill(scalar, s, 1);
	return *this;
}
