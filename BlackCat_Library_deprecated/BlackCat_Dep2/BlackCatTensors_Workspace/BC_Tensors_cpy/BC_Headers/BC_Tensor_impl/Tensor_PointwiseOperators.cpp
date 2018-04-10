#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type> Tensor<number_type>::operator^(const Tensor<number_type> & t) const {
    this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);
    Tensor_Operations<number_type>::power(store.tensor, tensor, t.tensor, sz);
    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator/(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);
    Tensor_Operations<number_type>::divide(store.tensor, tensor, t.tensor, sz);
    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator+(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);
    Tensor_Operations<number_type>::add(store.tensor, tensor, t.tensor, sz);
    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator-(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);
    Tensor_Operations<number_type>::subtract(store.tensor, tensor, t.tensor, sz);
    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator&(const Tensor<number_type>& t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);
    Tensor_Operations<number_type>::multiply(store.tensor, tensor, t.tensor, sz);
    return store;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator^=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);
    Tensor_Operations<number_type>::power(tensor, tensor, t.tensor, sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator/=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);
    Tensor_Operations<number_type>::divide(tensor, tensor, t.tensor, sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator+=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);
    Tensor_Operations<number_type>::add(tensor, tensor, t.tensor, sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator-=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);
    Tensor_Operations<number_type>::subtract(tensor, tensor, t.tensor, sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator&=(const Tensor<number_type>& t) {
	this->assert_same_dimensions(t);
    Tensor_Operations<number_type>::multiply(tensor, tensor, t.tensor, sz);
    return *this;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator-(const Scalar<number_type>& t) const {
    Tensor<number_type> s(*this, false);
    Tensor_Operations<number_type>::subtract(s.tensor, tensor, t(), sz);
    return s;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator&(const Scalar<number_type>& t) const {
    Tensor<number_type> s(*this, false);
    Tensor_Operations<number_type>::multiply(s.tensor, tensor, t(), sz);
    return s;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator^=(const Scalar<number_type>& t) {
    Tensor_Operations<number_type>::power(tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator/=(const Scalar<number_type>& t) {
    Tensor_Operations<number_type>::divide(tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator+=(const Scalar<number_type>& s) {
    Tensor_Operations<number_type>::add(tensor, tensor, s(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator-=(const  Scalar<number_type>&  t) {
    Tensor_Operations<number_type>::subtract(tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator&=(const  Scalar<number_type>&  t) {
    Tensor_Operations<number_type>::multiply(tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type> Tensor<number_type> Tensor<number_type>::operator^(const  Scalar<number_type>&  t) const {
    Tensor<number_type> s(*this, false);
    Tensor_Operations<number_type>::power(s.tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator/(const  Scalar<number_type>&  t) const {
    Tensor<number_type> s(*this, false);
    Tensor_Operations<number_type>::divide(s.tensor, tensor, t(), sz);
    return *this;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator+(const  Scalar<number_type>&  t) const {
    Tensor<number_type> s(*this, false);
    Tensor_Operations<number_type>::add(s.tensor, tensor, t(), sz);
    return *this;
}
