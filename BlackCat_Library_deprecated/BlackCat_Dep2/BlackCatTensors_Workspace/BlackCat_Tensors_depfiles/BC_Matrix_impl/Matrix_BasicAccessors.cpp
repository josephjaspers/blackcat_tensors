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
