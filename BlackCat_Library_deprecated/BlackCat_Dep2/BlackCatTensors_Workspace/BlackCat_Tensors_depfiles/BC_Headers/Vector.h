#ifndef VECTOR_H_
#define VECTOR_H_
#include "Matrix.h"

template<typename number_type>
class Vector : public Matrix<number_type> {

	template<typename mat_friend>
	friend class Matrix;

public:
	Vector<number_type>(const std::initializer_list<number_type>& vector);
	Vector<number_type>(const Vector<number_type>& t);
	Vector<number_type>(const Tensor<number_type>& t);
	Vector<number_type>(Vector<number_type>&& t);
	Vector<number_type>(Tensor<number_type>&& t);
	Vector<number_type>() : Matrix<number_type>() {/*empty*/}
	explicit Vector<number_type>(unsigned m) : Matrix<number_type>(m, 1) {/*empty*/}

	virtual ~Vector<number_type>() {};

	//OVERRIDES
	unsigned rows() const override;
	unsigned cols() const override;

	Vector<number_type>& operator=(const Vector<number_type>& t);
	Vector<number_type>& operator=(Vector<number_type>&&      t);
	Vector<number_type>& operator=(const Tensor<number_type>& t) override;
	Vector<number_type>& operator=(Tensor<number_type>&&      t) override;
	Vector<number_type>& operator=(const Scalar<number_type>& s) override;

	Matrix<number_type> operator->*(const Vector<number_type>& v) const {
		Matrix<number_type> r(this->size(), v.size());
		Tensor_Operations<number_type>::dot_outerproduct(r.data(), this->data(), this->size(),  v.data(), v.size());
		return r;
	}


	//void reshape(std::initializer_list<unsigned> new_shape) override { throw std::invalid_argument("cannot reshape tensor - convert to Tensor");};
	bool same_size(const Tensor<number_type>& t) const;
	void assert_same_size(const Tensor<number_type>& t) const;
};


#endif /* VECTOR_H_ */
