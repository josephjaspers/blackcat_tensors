#include "Scalar.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
template<typename number_type>


Scalar<number_type>::Scalar(const Scalar<number_type>& s) {
	Tensor_Operations<number_type>::initialize(scalar, 1);
	Tensor_Operations<number_type>::fill(scalar, s.scalar[0], 1);
}

template<typename number_type>
Scalar<number_type>::Scalar(Scalar<number_type> && s) {
	if (s.ownership && ownership)
	scalar = s.scalar;
	else {
		Tensor_Operations<number_type>::fill(scalar, s.scalar[0], 1);
	}
}

template<typename number_type>
Scalar<number_type>::Scalar(number_type value) {
	Tensor_Operations<number_type>::initialize(scalar, 1);
	Tensor_Operations<number_type>::fill(scalar, value, 1);
}

template<typename number_type>
Scalar<number_type>& Scalar<number_type>::operator =(const Scalar<number_type>& s) {
	Tensor_Operations<number_type>::fill(scalar, s.scalar[0], 1);
}

template<typename number_type>
Scalar<number_type>& Scalar<number_type>::operator =(Scalar<number_type> && s) {

	if (s.ownership && ownership) {
		Tensor_Operations<number_type>::destruction(this->scalar);
		this->scalar = s.scalar;
	} else {
		Tensor_Operations<number_type>::fill(scalar, s.scalar[0], 1);
	}
}

template<typename number_type>
Scalar<number_type>& Scalar<number_type>::operator =(number_type s) {
	Tensor_Operations<number_type>::fill(scalar, s, 1);
	return *this;
}
