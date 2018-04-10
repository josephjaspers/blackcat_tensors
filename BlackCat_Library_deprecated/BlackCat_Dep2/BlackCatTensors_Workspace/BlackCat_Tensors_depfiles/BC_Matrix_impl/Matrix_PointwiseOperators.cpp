//#include "Matrix.h"
//#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
//
//
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator^(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::power(store.data(), this->data(), t.data(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator/(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::divide(store.data(), this->data(), t.data(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator+(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::add(store.data(), this->data(), t.data(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator-(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::subtract(store.data(), this->data(), t.data(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator&(const Tensor<number_type>& m) const {
//		this->assert_same_dimensions(m);
//
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::multiply(store.data(), this->data(), m.data(), this->size());
//		return store;
//	}
//
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator^(const Scalar<number_type>& t) const {
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::power(store.data(), this->data(), t(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator/(const Scalar<number_type>& t) const {
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::divide(store.data(), this->data(), t(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator+(const Scalar<number_type>& t) const {
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::add(store.data(), this->data(), t(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator-(const Scalar<number_type>& t) const {
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::subtract(store.data(), this->data(), t(), this->size());
//		return store;
//	}
//	template<typename number_type> Matrix<number_type> Matrix<number_type>::operator&(const Scalar<number_type>& t) const {
//		Matrix<number_type> store(this->rows(), this->cols());
//		Tensor_Operations<number_type>::multiply(store.data(), this->data(), t(), this->size());
//		return store;
//	}
