#ifndef VECTOR_H_
#define VECTOR_H_
#include "Matrix.h"

template<typename number_type, class TensorOperations = CPU>
class Vector : public Matrix<number_type, TensorOperations> {

public:
	Vector<number_type, TensorOperations>(const std::initializer_list<number_type>& vector);
	Vector<number_type, TensorOperations>(const Vector<number_type, TensorOperations>& t);
	Vector<number_type, TensorOperations>(const Tensor<number_type, TensorOperations>& t);
	Vector<number_type, TensorOperations>(Vector<number_type, TensorOperations>&& t);
	Vector<number_type, TensorOperations>(Tensor<number_type, TensorOperations>&& t);
	Vector<number_type, TensorOperations>() : Matrix<number_type, TensorOperations>() {/*empty*/}
	explicit Vector<number_type, TensorOperations>(unsigned m) : Matrix<number_type, TensorOperations>(m, 1) {/*empty*/}

	virtual ~Vector<number_type, TensorOperations>() {};

	Vector<number_type, TensorOperations>& operator=(const Vector<number_type, TensorOperations>& t);
	Vector<number_type, TensorOperations>& operator=(Vector<number_type, TensorOperations>&&      t);
	Vector<number_type, TensorOperations>& operator=(const Tensor<number_type, TensorOperations>& t) override;
	Vector<number_type, TensorOperations>& operator=(Tensor<number_type, TensorOperations>&&      t) override;
	Vector<number_type, TensorOperations>& operator=(const Scalar<number_type, TensorOperations>& s) override;
	Vector<number_type, TensorOperations>& operator=(std::initializer_list<number_type> s) override { this->Tensor<number_type, TensorOperations>::operator=(s); return * this;}
	Vector<number_type, TensorOperations>& operator=(number_type s) override { this->Tensor<number_type, TensorOperations>::operator=(s); return * this;}

};


#endif /* VECTOR_H_ */
