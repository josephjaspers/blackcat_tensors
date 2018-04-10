/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include "Tensor.h"

template<typename number_type> Tensor<number_type>& Tensor<number_type>::operator=(const Scalar<number_type>& val) {
    if (tensor) {
        Tensor_Operations<number_type>::fill(tensor, val(), sz);
    }
    return *this;
}

template<typename number_type> inline Tensor<number_type>& Tensor<number_type>::operator=(const Tensor<number_type>& cpy) {
    if (this->isInitialized()) {
        this->assert_same_dimensions(cpy);
        Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);
    } else {
        Tensor_Operations<number_type>::initialize(tensor, cpy.size());
        Tensor_Operations<number_type>::copy(tensor, cpy.tensor, cpy.size());
        sz = cpy.sz;
        order = cpy.order;

        ranks = new unsigned[order];
        BC::copy(ranks, cpy.ranks, order);
    }
    return *this;
}

template<typename number_type> Tensor<number_type>& Tensor<number_type>::operator=(Tensor<number_type>&& cpy) {

	if (isInitialized()) {
		assert_same_dimensions(cpy);

		if (ownership) {
			if (cpy.ownership) {
				Tensor_Operations<number_type>::destruction(tensor);
				tensor = cpy.tensor;

				delete[] ranks;
				ranks = cpy.ranks;

				cpy.reset_post_move();
			}
		} else {
			Tensor_Operations<number_type>::copy(this->tensor, cpy.tensor, sz);
		}
	} else {
		sz = cpy.sz;
		order = cpy.order;

		if (cpy.ownership) {
			tensor = cpy.tensor;
			ranks = cpy.ranks;

			cpy.reset_post_move();
		} else {
	        Tensor_Operations<number_type>::initialize(tensor, sz);
	        Tensor_Operations<number_type>::copy(tensor, cpy.tensor, sz);
	        ranks = new unsigned[order];
	        BC::copy(ranks, cpy.ranks, order);
		}
	}

    return*this;
}

template<typename number_type>
Tensor<number_type>& Tensor<number_type>::operator = (std::initializer_list<number_type> vector) {

	if (isInitialized()){
		if (sz != vector.size()) {
			throw std::invalid_argument("operator =(initializer_list<number_type>)  Error: sz mismatch");
		}
		Tensor_Operations<number_type>::copy(tensor, vector.begin(), sz);
	} else {
	sz = vector.size();
	ranks = new unsigned[1];
	ranks[0] = 1;
	order = 1;
	Tensor_Operations<number_type>::initialize(tensor, sz);
	Tensor_Operations<number_type>::copy(tensor, vector.begin(), sz);
	}
	return *this;
}
