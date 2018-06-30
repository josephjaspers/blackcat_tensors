/*
 * alias.h
 *
 *  Created on: Jun 28, 2018
 *      Author: joseph
 */

#ifndef ALIAS_H_
#define ALIAS_H_

namespace BC {
namespace Base {

template<class> class Tensor_Operations;

template<class derived>
struct Alias {


	using mathlib_type = _mathlib<derived>;

	derived& tensor;

	Alias(derived& tensor_) : tensor(tensor_) {}

	template<class derived_t>
	void evaluate(const Tensor_Operations<derived_t>& param) {
		BC::Evaluator<mathlib_type>::evaluate_aliased(tensor.internal(), static_cast<const derived_t&>(param).internal());
	}

	template<class derived_t>
	auto& operator = (const Tensor_Operations<derived_t>& param) {
		tensor.assert_valid(param);
		evaluate(tensor.bi_expr(oper::assign(), param));
		return tensor;
	}

	template<class derived_t>
	auto& operator += (const Tensor_Operations<derived_t>& param) {
		tensor.assert_valid(param);
		evaluate(tensor.bi_expr(oper::add_assign(), param));
		return tensor.as_derived();
	}

	template<class derived_t>
	auto& operator -= (const Tensor_Operations<derived_t>& param) {
		tensor.assert_valid(param);
		evaluate(tensor.bi_expr(oper::sub_assign(), param));
		return tensor.as_derived();
	}
};





}
}

#endif /* ALIAS_H_ */
