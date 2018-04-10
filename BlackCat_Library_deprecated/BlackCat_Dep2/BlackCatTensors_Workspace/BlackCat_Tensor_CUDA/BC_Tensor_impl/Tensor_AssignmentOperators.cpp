/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include "Tensor.h"

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator=(const Scalar<number_type, TensorOperations>& val) {

	CPU::fill(tensor, ranks, order, ld, val());
    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>
inline Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator=(const Tensor<number_type, TensorOperations>& cpy) {
    if (this->isInitialized()) {
        this->assert_same_dimensions(cpy);
        TensorOperations::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld);
    } else {
    	TensorOperations::initialize(tensor, cpy.size());
        order = cpy.order;
        TensorOperations::unified_initialize(ranks, order);
        BC::copy(ranks, cpy.ranks, order);
        TensorOperations::unified_initialize(ld, order);
        BC::init_leading_dimensions(ld, ranks, order);
        sz = cpy.size();

        TensorOperations::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld);
    }
    alertUpdate();
    return *this;
}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator=(Tensor<number_type, TensorOperations>&& cpy) {


	if (isInitialized()) {
		assert_same_dimensions(cpy);

		if (tensor_ownership) {
			if (cpy.tensor_ownership) {
				TensorOperations::destruction(tensor);
				tensor = cpy.tensor;

				delete[] ranks;
				ranks = cpy.ranks;

				cpy.reset_post_move();
			}
		} else {
			TensorOperations::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld);
		}
	} else {
		sz = cpy.sz;
		order = cpy.order;

		if (cpy.tensor_ownership) {
			tensor = cpy.tensor;
			ranks = cpy.ranks;
			ld = cpy.ld;

			cpy.reset_post_move();
		} else {
			TensorOperations::initialize(tensor, size());
	        TensorOperations::unified_initialize(ranks, order);
	        BC::copy(ranks, cpy.ranks, order);
	        TensorOperations::unified_initialize(ld, order);
	        BC::init_leading_dimensions(ld, ranks, order);
	        TensorOperations::copy(tensor, ranks, order, ld, cpy.tensor, cpy.ld);
		}
	}

    alertUpdate();
    return*this;
}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::operator= (std::initializer_list<number_type> vector) {

	if (isInitialized()){
		if (size() != vector.size()) {
			throw std::invalid_argument("operator =(initializer_list<number_type>)  Error: sz mismatch");
		}
		if (this->subTensor) {
			unsigned * ld_tmp;
		    TensorOperations::unified_initialize(ld_tmp, order);
			BC::init_leading_dimensions(ld_tmp, ranks, order);
			TensorOperations::copy(tensor, ranks, order, ld, vector.begin(), ld_tmp);
		    TensorOperations::destruction(ld_tmp);
		} else {
			TensorOperations::copy(tensor, vector.begin(), size());
		}
	} else {
    TensorOperations::unified_initialize(ranks, 1);
	ranks[0] = vector.size();
    TensorOperations::unified_initialize(ld, order);
	ld[0] = 1;
	order = 1;
	sz = vector.size();

	TensorOperations::initialize(tensor, size());
	TensorOperations::copy(tensor, vector.begin(), size());
	}

    alertUpdate();
	return *this;
}
