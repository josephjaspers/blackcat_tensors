#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"
#include "Tensor.h"

template<typename number_type, class TensorOperations>
void Tensor<number_type, TensorOperations>::reset_post_move() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif
	sz = 0;
	tensor = nullptr;
    ranks = nullptr;
    ld = nullptr;
    //delete self_transposed;
    //self_transposed = nullptr;
	sz = 0;
    order = 0;
    clearTensorCache();
    alertUpdate();
}

template<typename number_type, class TensorOperations>
void Tensor<number_type, TensorOperations>::reset() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif

   CPU::destruction(tensor);
   delete[] ranks;
   delete[] ld;
 //  delete self_transposed;

    sz = 0;
    tensor = nullptr;
    ranks = nullptr;
    ld = nullptr;
  //  self_transposed = nullptr;
    order = 0;

    clearTensorCache();
    alertUpdate();
}

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::reshape(Shape new_shape) {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
	unsigned new_sz = 1;
    for (auto iter = new_shape.begin(); iter != new_shape.end(); ++iter) {
        new_sz *= *iter;
    }

    if (size() != new_sz) {
        throw std::invalid_argument("reshape sz mismatch");
    }
    #endif


    delete[] ld;
    delete[] ranks;
    order = new_shape.size();
    ranks = new unsigned[order];
    ld = new unsigned[order];

    BC::copy(ranks, &new_shape[0], order);
	BC::init_leading_dimensions(ld, ranks, order);

    clearTensorCache();
    alertUpdate();
    return * this;
}
template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>& Tensor<number_type, TensorOperations>::flatten() {
#ifndef BLACKCAT_DISABLE_ADVANCED_CHECKS
	if (!tensor_ownership) {
		throw std::invalid_argument("ID_Tensor reset -- cannot reset a subtensor");
	}
#endif
	reshape({size(), 1});
	order = 1;
    alertUpdate();
    return * this;
}

