//
//#include "Vector.h"
//#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
//
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator^(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::power(store.tensor, this->tensor, t.tensor, this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator/(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::divide(store.tensor, this->tensor, t.tensor, this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator+(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::add(store.tensor, this->tensor, t.tensor, this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator-(const Tensor<number_type>& t) const {
//		this->assert_same_dimensions(t);
//
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::subtract(store.tensor, this->tensor, t.tensor, this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator&(const Tensor<number_type>& m) const {
//		this->assert_same_dimensions(m);
//
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::multiply(store.tensor, this->tensor, m.tensor, this->size());
//		return store;
//	}
//
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator^(const Scalar<number_type>& t) const {
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::power(store.tensor, this->tensor, t(), this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator/(const Scalar<number_type>& t) const {
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::divide(store.tensor, this->tensor, t(), this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator+(const Scalar<number_type>& t) const {
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::add(store.tensor, this->tensor, t(), this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator-(const Scalar<number_type>& t) const {
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::subtract(store.tensor, this->tensor, t(), this->size());
//		return store;
//	}
//	template<typename number_type> Vector<number_type> Vector<number_type>::operator&(const Scalar<number_type>& t) const {
//		Vector<number_type> store(this->size());
//		Tensor_Operations<number_type>::multiply(store.tensor, this->tensor, t(), this->size());
//		return store;
//	}
