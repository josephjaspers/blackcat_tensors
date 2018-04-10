#ifndef VECTOR_H_
#define VECTOR_H_
#include "Matrix.h"

template<typename number_type>
class Vector : public Matrix<number_type> {

public:
	Vector<number_type>(const std::initializer_list<number_type>& vector);
	Vector<number_type>(const Vector<number_type>& t);
	Vector<number_type>(const Tensor<number_type>& t);
	Vector<number_type>(Vector<number_type>&& t);
	Vector<number_type>(Tensor<number_type>&& t);
	Vector<number_type>() : Matrix<number_type>() {/*empty*/}
	explicit Vector<number_type>(unsigned m) : Matrix<number_type>(m, 1) {/*empty*/}

	virtual ~Vector<number_type>() {};

	Vector<number_type>& operator=(const Vector<number_type>& t);
	Vector<number_type>& operator=(Vector<number_type>&&      t);
	Vector<number_type>& operator=(const Tensor<number_type>& t) override;
	Vector<number_type>& operator=(Tensor<number_type>&&      t) override;
	Vector<number_type>& operator=(const Scalar<number_type>& s) override;
	Vector<number_type>& operator=(std::initializer_list<number_type> s) override { this->Tensor<number_type>::operator=(s); return * this;}

};


#endif /* VECTOR_H_ */
