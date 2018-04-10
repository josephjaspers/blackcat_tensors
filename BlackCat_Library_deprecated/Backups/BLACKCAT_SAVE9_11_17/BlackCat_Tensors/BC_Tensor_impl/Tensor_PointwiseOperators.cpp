#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type>
Tensor<number_type> Tensor<number_type>::operator^(const Tensor<number_type> & t) const {
    this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);

    subTensor || t.subTensor ?
    		Tensor_Operations<number_type>::power(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld)
    		:
    		Tensor_Operations<number_type>::power(store.tensor, tensor, t.tensor, size());


    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator/(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);

    subTensor || t.subTensor ?
        		Tensor_Operations<number_type>::divide(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld)
        		:
        		Tensor_Operations<number_type>::divide(store.tensor, tensor, t.tensor, size());

    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator+(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);

    subTensor || t.subTensor ?
          		Tensor_Operations<number_type>::add(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld)
          		:
          		Tensor_Operations<number_type>::add(store.tensor, tensor, t.tensor, size());


    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator-(const Tensor<number_type> & t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);

    subTensor || t.subTensor ?
           		Tensor_Operations<number_type>::subtract(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld)
           		:
           		Tensor_Operations<number_type>::subtract(store.tensor, tensor, t.tensor, size());


    return store;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator&(const Tensor<number_type>& t) const {
	this->assert_same_dimensions(t);
    Tensor<number_type> store(*this, false);

    subTensor || t.subTensor ?
             		Tensor_Operations<number_type>::multiply(store.tensor, ranks, order, store.ld, tensor, ld, t.tensor, t.ld)
             		:
             		Tensor_Operations<number_type>::multiply(store.tensor, tensor, t.tensor, size());


    return store;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator^=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);

	subTensor || t.subTensor ?
	    Tensor_Operations<number_type>::power(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld)
	              		:
		Tensor_Operations<number_type>::power(tensor, tensor, t.tensor, size());
    alertUpdate();
	return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator/=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);

	subTensor || t.subTensor ?
      		Tensor_Operations<number_type>::divide(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld)
		:
      		Tensor_Operations<number_type>::divide(tensor, tensor, t.tensor, size());


    alertUpdate();
	return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator+=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);

	subTensor || t.subTensor ?
      		Tensor_Operations<number_type>::add(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld)
      		:
			Tensor_Operations<number_type>::add(tensor, tensor, t.tensor, size());


    alertUpdate();
	return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator-=(const Tensor<number_type> & t) {
	this->assert_same_dimensions(t);

	subTensor || t.subTensor ?
      		Tensor_Operations<number_type>::subtract(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld)
      		:
		    Tensor_Operations<number_type>::subtract(tensor, tensor, t.tensor, size());


    alertUpdate();
	return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator&=(const Tensor<number_type>& t) {
	this->assert_same_dimensions(t);

	subTensor || t.subTensor ?
      		Tensor_Operations<number_type>::multiply(tensor, ranks, order, ld, tensor, ld, t.tensor, t.ld)
      			:
			Tensor_Operations<number_type>::multiply(tensor, tensor, t.tensor, size());

    alertUpdate();
	return *this;
}
//------------------------------------------------------------scalar methods------------------------------------------------------------//

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator^=(const Scalar<number_type>& scal) {
    subTensor ?
    	    Tensor_Operations<number_type>::power(tensor, ranks, order, ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::power(tensor, tensor, scal(), size());
    alertUpdate();
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator/=(const Scalar<number_type>& scal) {
    subTensor ?
    	    Tensor_Operations<number_type>::divide(tensor, ranks, order, ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::divide(tensor, tensor, scal(), size());
    alertUpdate();
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator+=(const Scalar<number_type>& scal) {
    subTensor ?
    	    Tensor_Operations<number_type>::add(tensor, ranks, order, ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::add(tensor, tensor, scal(), size());
    alertUpdate();
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator-=(const  Scalar<number_type>&  scal) {
    subTensor ?
    	    Tensor_Operations<number_type>::subtract(tensor, ranks, order, ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::subtract(tensor, tensor, scal(), size());
    alertUpdate();
    return *this;
}

template<typename number_type>Tensor<number_type>& Tensor<number_type>::operator&=(const  Scalar<number_type>&  scal) {
    subTensor ?
    	    Tensor_Operations<number_type>::multiply(tensor, ranks, order, ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::multiply(tensor, tensor, scal(), size());
    alertUpdate();
    return *this;
}
//--------------------------------scalar non assignment --------------------------------------------------------//

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator-(const Scalar<number_type>& scal) const {
    Tensor<number_type> s(*this, false);

    subTensor ?
    	    Tensor_Operations<number_type>::subtract(s.tensor, s.ranks, s.order, s.ld, tensor, ld, scal())
    		:
    Tensor_Operations<number_type>::subtract(s.tensor, tensor, scal(), size());
    return s;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator&(const Scalar<number_type>& scal) const {
    Tensor<number_type> s(*this, false);

    subTensor ?
    	    Tensor_Operations<number_type>::multiply(s.tensor, ranks, order, s.ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::multiply(s.tensor, tensor, scal(), size());
    return s;
}

template<typename number_type> Tensor<number_type> Tensor<number_type>::operator^(const  Scalar<number_type>&  scal) const {
    Tensor<number_type> s(*this, false);

    subTensor ?
    	    Tensor_Operations<number_type>::power(s.tensor, ranks, order, s.ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::power(s.tensor, tensor, scal(), size());
    return s;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator/(const  Scalar<number_type>&  scal) const {
    Tensor<number_type> s(*this, false);

    subTensor ?
    	    Tensor_Operations<number_type>::divide(s.tensor, ranks, order, s.ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::divide(s.tensor, tensor, scal(), size());
    return s;
}

template<typename number_type>Tensor<number_type> Tensor<number_type>::operator+(const  Scalar<number_type>&  scal) const {
    Tensor<number_type> s(*this, false);

    subTensor ?
    	    Tensor_Operations<number_type>::add(s.tensor, ranks, order, s.ld, tensor, ld, scal())
    		:
    	    Tensor_Operations<number_type>::add(s.tensor, tensor, scal(), size());
    return s;
}
