/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_TENSOR_EXPRESSION_TEMPLATES_INJECTION_H_
#define BC_TENSOR_EXPRESSION_TEMPLATES_INJECTION_H_


namespace BC {
namespace tensors {
namespace exprs {

template<class Tensor, int AlphaModifer=1, int BetaModifer=0>
struct Output_Data {

	static constexpr size_t ALPHA = AlphaModifer;
	static constexpr size_t BETA = BetaModifer;

	Tensor array;
	const Tensor& data() const { return array; }
		  Tensor& data()       { return array; }
};

template<class Op, bool PriorEval, class Tensor, int A, int B>
auto update_alpha_beta_modifiers(Output_Data<Tensor, A, B> tensor) {
	constexpr int alpha =  A * BC::oper::operation_traits<Op>::alpha_modifier;
	constexpr int beta  = PriorEval ? 1 : 0;
	return Output_Data<Tensor, alpha, beta> {tensor.data()};
}

template<int AlphaModifer=1, int BetaModifer=0, class Tensor>
auto make_output_data(Tensor tensor) {
	return Output_Data<Tensor, AlphaModifer, BetaModifer> { tensor };
}


} //ns exprs
} //ns tensors
} //ns BC


#endif /* INJECTION_INFO_H_ */
