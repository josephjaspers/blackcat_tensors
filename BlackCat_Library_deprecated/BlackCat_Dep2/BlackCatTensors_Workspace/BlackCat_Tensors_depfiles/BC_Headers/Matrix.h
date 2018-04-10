#ifndef BLACKCAT_Matrix_h
#define BLACKCAT_Matrix_h

#include "Tensor.h"

template <typename number_type>
class Matrix : public Tensor<number_type> {

public:
	Matrix<number_type>(const Tensor<number_type>& t) : Tensor<number_type>(t) { this->assert_isMatrix(*this); };
	Matrix<number_type>(Tensor<number_type>&& t) : Tensor<number_type>(t) { this->assert_isMatrix(*this); };

	Matrix& operator=(const Tensor<number_type>& t) {this->Tensor<number_type>::operator=(t); return * this;};
	Matrix& operator=(Tensor<number_type>&& t) {this->Tensor<number_type>::operator=(t); return * this;};
	Matrix& operator=(const Scalar<number_type>& s) {this->Tensor<number_type>::operator=(s); return * this;};

	Matrix<number_type>() : Tensor<number_type>() {/*empty*/}
	explicit Matrix<number_type>(unsigned m, unsigned n) : Tensor<number_type>(m, n) {/*empty*/}
	~Matrix<number_type>() {};

	void reshape(std::initializer_list<unsigned> new_shape) override;
	bool same_dimensions(const Tensor<number_type>& t) const override;
	bool dotProduct_dimensions(const Tensor<number_type>& t) const override;
};

#endif
