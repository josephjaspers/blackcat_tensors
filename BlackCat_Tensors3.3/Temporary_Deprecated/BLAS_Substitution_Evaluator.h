/*
 * BLAS_Substitution_Evaluator.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef BLAS_SUBSTITUTION_EVALUATOR_H_
#define BLAS_SUBSTITUTION_EVALUATOR_H_

namespace BC{


namespace internal {

	static constexpr bool INJECTION();

	//shorthands
	template<class lv, class rv, class oper> using be = binary_expression<lv, rv, oper>;
	template<class v, class oper> using ue = unary_expression<v, oper>;


	//NO SUBSTITUTION
	template<class T, class enabler = void> struct substituter {
		static constexpr bool conditional = false;
		static constexpr int priority = - 1;

		template<class injection_type> using type = T;

		using substitution_t = void; //no sub type
	};

	//unary_expression
	template<class V, class func>
	struct substituter<ue<V, func>> {
		static constexpr bool conditional = substituter<V>::conditional;

		template<class injection_type>
		using type = ue<typename injector<V>::template type<injection_type>, func>;

		using substitution_t = typename substituter<V>::substitution_t;
	};

	//binary_expression
	template<class lv, class rv, class func>
	struct substituter<be<lv, rv, func>> {
		static constexpr bool lv_branch = substituter<lv>::conditional;
		static constexpr bool rv_branch = substituter<rv>::conditional;
		static constexpr bool conditional = !INJECTION<be<lv,rv,func>> && (lv_branch || rv_branch);

		//injection is always a core
		template<class injection_type>
		using type = be<typename substituter<lv>::template type<injection_type>, be<typename substituter<rv>::template type<injection_type>>>;
	};

	template<class lv, class rv, class ml>
	struct substituter<be<lv, rv, BC::oper::dotproduct<ml>>, void> {
		static constexpr int priority = -1;
		static constexpr bool conditional = true;

		//injection_type is the "core" injection (slice/chunk/row/core) this causes a "drop-in" replacement for the blas-function
		template<class injection_type> using type = std::decay_t<injection_type>;
	};

}

}



#endif /* BLAS_SUBSTITUTION_EVALUATOR_H_ */
