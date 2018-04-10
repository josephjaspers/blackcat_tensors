#include "Tensor.h"
#include "BLACKCAT_EXPLICIT_INSTANTIATION.h"

template<typename number_type, class TensorOperations>
Tensor<number_type, TensorOperations>::Tensor(const Tensor<number_type, TensorOperations>& m1, const Tensor<number_type, TensorOperations>& m2) :
tensor_ownership(true), rank_ownership(true), ld_ownership(true), subTensor(false) {
	//DOTPRODUCT CONSTRUCTOR
	order = m2.order;
	TensorOperations::unified_initialize(ranks, order);
	BC::copy(ranks, m2.ranks, order);

	ranks[0] = m1.rows();
	if (order > 1)
	ranks[1] = m2.cols();

	TensorOperations::unified_initialize(ld, order);
	BC::init_leading_dimensions(ld, ranks, order);
	sz = BC::calc_sz(ranks, order);

	TensorOperations::initialize(tensor, size());
}

