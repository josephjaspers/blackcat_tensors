#ifndef BLACKCAT_Matrix_h
#define BLACKCAT_Matrix_h

#include "Tensor.h"

template <typename number_type>
class Matrix : public Tensor<number_type> {

public:
	Matrix<number_type>(const Tensor<number_type>& t) 	: Tensor<number_type>(t) { this->assert_isMatrix(*this); };
	Matrix<number_type>(Tensor<number_type>&& t) 		: Tensor<number_type>(t) { this->assert_isMatrix(*this); };

	Matrix<number_type>& operator=(const Tensor<number_type>& t)			override {this->Tensor<number_type>::operator=(t); return * this;};
	Matrix<number_type>& operator=(Tensor<number_type>&& t) 				override {this->Tensor<number_type>::operator=(t); return * this;};
	Matrix<number_type>& operator=(const Scalar<number_type>& s)			override {this->Tensor<number_type>::operator=(s); return * this;};
	Matrix<number_type>& operator=(std::initializer_list<number_type> v) 	override {this->Tensor<number_type>::operator=(v); return * this;}

	Matrix<number_type>() : Tensor<number_type>() {/*empty*/}
	explicit Matrix<number_type>(unsigned m, unsigned n) : Tensor<number_type>(m, n) {/*empty*/}
	~Matrix<number_type>() {};

	bool same_dimensions(const Tensor<number_type>& t) 			const override;
	bool dotProduct_dimensions(const Tensor<number_type>& t) 	const override;
	Tensor<number_type>& reshape(std::initializer_list<unsigned> new_shape) override { this->Tensor<number_type>::reshape(new_shape); this->assert_isMatrix(*this); return *this;};
};

#endif
