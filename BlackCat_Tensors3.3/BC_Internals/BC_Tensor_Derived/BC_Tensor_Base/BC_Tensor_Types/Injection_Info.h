/*
 * Injection_Info.h
 *
 *  Created on: Jun 9, 2018
 *      Author: joseph
 */

#ifndef INJECTION_INFO_H_
#define INJECTION_INFO_H_

namespace BC {
namespace internal {

/*
 * class utilized for injections for BLAS -- wraps the injected_tensor and carries the relevant scalar modifier
 */

template<class tensor_core, class mathlib, int alpha_modifier, int beta_modifier>
struct injection_wrapper {

	using scalar = _scalar<tensor_core>;

	tensor_core& array;

	static constexpr bool alpha_abnormal = alpha_modifier != 1;
	static constexpr bool beta_abnormal  = beta_modifier != 0;

	static const scalar* alpha_mod = mathlib::static_initialize(1, alpha_modifier);
	static const scalar* beta_mod = mathlib::static_initialize(1, beta_modifier);


	auto& internal() const { return array; }
	auto& internal()	   { return array; }

	auto*& alpha() const { return alpha_mod; }
	auto*& beta()	   	 { return beta_mod; }

	constexpr bool injected_alpha() { return alpha_abnormal; }
	constexpr bool injected_beta()  { return beta_abnormal; }
};

}



#endif /* INJECTION_INFO_H_ */
