#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator^(const Tensor<number_type, TensorOperations> & t) const {
    this->assert_same_dimensions(t);
    Tensor<number_type, TensorOperations> store(*this, false);

    TensorOperations::power(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld);

    return store;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator/(const Tensor<number_type, TensorOperations> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type, TensorOperations> store(*this, false);

    TensorOperations::divide(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld);

    return store;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator+(const Tensor<number_type, TensorOperations> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type, TensorOperations> store(*this, false);

    TensorOperations::add(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld);

    return store;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator-(const Tensor<number_type, TensorOperations> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type, TensorOperations> store(*this, false);

    TensorOperations::subtract(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld);

    return store;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator&(const Tensor<number_type, TensorOperations>& t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type, TensorOperations> store(*this, false);

    TensorOperations::multiply(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld);

    return store;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator^=(const Tensor<number_type, TensorOperations> & t) {
	this->assert_same_dimensions(t);

	TensorOperations::power(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld);

	alertUpdate();
	return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator/=(const Tensor<number_type, TensorOperations> & t) {
	this->assert_same_dimensions(t);

	TensorOperations::divide(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld);

    alertUpdate();
	return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator+=(const Tensor<number_type, TensorOperations> & t) {
	this->assert_same_dimensions(t);

	TensorOperations::add(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld);

    alertUpdate();
	return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator-=(const Tensor<number_type, TensorOperations> & t) {
	this->assert_same_dimensions(t);

	TensorOperations::subtract(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld);

    alertUpdate();
	return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator&=(const Tensor<number_type, TensorOperations>& t) {
	this->assert_same_dimensions(t);

	TensorOperations::multiply(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld);

    alertUpdate();
	return *this;
}
//------------------------------------------------------------scalar methods------------------------------------------------------------//

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator^=(const Scalar<number_type, TensorOperations>& scal) {
	TensorOperations::power(tensor, ranks, order, ld, tensor, ld, scal());

	alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator/=(const Scalar<number_type, TensorOperations>& scal) {
 	TensorOperations::divide(tensor, ranks, order, ld, tensor, ld, scal());

 	alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator+=(const Scalar<number_type, TensorOperations>& scal) {
	TensorOperations::add(tensor, ranks, order, ld, tensor, ld, scal());

    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator-=(const  Scalar<number_type, TensorOperations>&  scal) {
	TensorOperations::subtract(tensor, ranks, order, ld, tensor, ld, scal());

    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator&=(const  Scalar<number_type, TensorOperations>&  scal) {

	TensorOperations::multiply(tensor, ranks, order, ld, tensor, ld, scal());

    alertUpdate();
    return *this;
}
//--------------------------------scalar non assignment --------------------------------------------------------//

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator-(const Scalar<number_type, TensorOperations>& scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::subtract(s.tensor, s.ranks, s.order, s.ld, tensor, ld, scal());

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator&(const Scalar<number_type, TensorOperations>& scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::multiply(s.tensor, ranks, order, s.ld, tensor, ld, scal());

    return s;
}

template<typename number_type, class TensorOperations> Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator^(const  Scalar<number_type, TensorOperations>&  scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::power(s.tensor, ranks, order, s.ld, tensor, ld, scal());

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator/(const  Scalar<number_type, TensorOperations>&  scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::divide(s.tensor, ranks, order, s.ld, tensor, ld, scal());

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator+(const  Scalar<number_type, TensorOperations>&  scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::add(s.tensor, ranks, order, s.ld, tensor, ld, scal());

    return s;
}

//--------------------------------------------------------------base number-type ------------------------------------------------------------/


template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator^=(number_type scal) {
	TensorOperations::power(tensor, ranks, order, ld, tensor, ld, scal);

	alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator/=(number_type scal) {
 	TensorOperations::divide(tensor, ranks, order, ld, tensor, ld, scal);

 	alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator+=(number_type scal) {
	TensorOperations::add(tensor, ranks, order, ld, tensor, ld, scal);

    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator-=(number_type scal) {
	TensorOperations::subtract(tensor, ranks, order, ld, tensor, ld, scal);

    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator&=(number_type scal) {

	TensorOperations::multiply(tensor, ranks, order, ld, tensor, ld, scal);

    alertUpdate();
    return *this;
}
//--------------------------------scalar non assignment --------------------------------------------------------//

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator=(number_type scal) {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::fill(tensor, ranks, order, ld, scal);

    return *this;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator-(number_type scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::subtract(s.tensor, s.ranks, s.order, s.ld, tensor, ld, scal);

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator&(number_type scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::multiply(s.tensor, ranks, order, s.ld, tensor, ld, scal);

    return s;
}

template<typename number_type, class TensorOperations> Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator^(number_type scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::power(s.tensor, ranks, order, s.ld, tensor, ld, scal);

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator/(number_type scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::divide(s.tensor, ranks, order, s.ld, tensor, ld, scal);

    return s;
}

template<typename number_type, class TensorOperations>Tensor<number_type, TensorOperations> Tensor<number_type, TensorOperations>::operator+(number_type scal) const {
    Tensor<number_type, TensorOperations> s(*this, false);
    TensorOperations::add(s.tensor, ranks, order, s.ld, tensor, ld, scal);

    return s;
}
