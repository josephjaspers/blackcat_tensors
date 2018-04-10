#include "Matrix.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"


template<typename number_type>
bool Matrix<number_type>::same_dimensions(const Tensor<number_type>& t) const {
	return t.degree() <= 2 ? this->rows() == t.rows() && this->cols() == t.cols() : false;
}

template<typename number_type>
bool Matrix<number_type>::dotProduct_dimensions(const Tensor<number_type>& t) const {
	return t.degree() <= 2 ? this->cols() == t.rows() : false;
}


//template<typename number_type>
//Tensor<number_type>& Matrix<number_type>::reshape(std::initializer_list<unsigned> new_shape) {
//	if (new_shape.size() != 2) {
//		throw std::invalid_argument("illegal conversion of Matrix to non-matrix shape");
//	}
//	this->Tensor<number_type>::reshape(new_shape);
//	return*this;
//}
