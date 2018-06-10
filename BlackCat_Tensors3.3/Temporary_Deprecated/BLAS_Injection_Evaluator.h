/*
 * BLAS_Injection_Evaluator.h
 *
 *  Created on: Jun 4, 2018
 *      Author: joseph
 */

#ifndef BLAS_INJECTION_EVALUATOR_H_
#define BLAS_INJECTION_EVALUATOR_H_

#include <type_traits>
namespace BC {


namespace oper {

//priority -1 == non-rotateable (injection not legal)
//priority 0 == low
//priority 1 == high
/*
 * a low priority may be after a high priority (ensure priority order)
 * but a low priority preceding a high-priority operation will result in non-injectable segment
 *
 * This module assigns arithmetic values to operands to handle apropriate precedence in injections
 */
template<class> struct dotproduct;
template<class> struct PRIORITY { static constexpr int value = -1;};
template<> struct PRIORITY<add> { static constexpr int value = 0; };
template<> struct PRIORITY<sub> { static constexpr int value = 0; };
template<> struct PRIORITY<mul> { static constexpr int value = 1; };
template<> struct PRIORITY<div> { static constexpr int value = 1; };
template<> struct PRIORITY<add_assign> { static constexpr int value = 0; };
template<> struct PRIORITY<sub_assign> { static constexpr int value = 0; }; //this will eventually be allowed but for now NO
template<> struct PRIORITY<assign> { static constexpr int value = 1; };

}

namespace internal {
	//DEFAULT -- (non-BLAS injection on unknown/non-specified functions)
	template<class T, class enabler = void> struct injector {
		static constexpr bool conditional = false;
		static constexpr int priority = -1;
		template<class injection_type> using type = T;
	};
	//ALL UNARY FUNCTIONS HAVE HIGH_PRIORITY
	template<class V, class func>
	struct injector<unary_expression<V, func>> {
		static constexpr int priority = 1;
		static constexpr bool conditional = injector<V>::priority <= priority && injector<V>::conditional;

		template<class injection_type> using type = typename injector<V>::template type<injection_type>;
	};
	//BINARY EXPRESSION - detect precedence of the func (operand),
	template<class lv, class rv, class func>
	struct injector<binary_expression<lv, rv, func>> {
		static constexpr int priority = oper::PRIORITY<func>::value;
		static constexpr bool lv_branch = injector<lv>::priority <= priority && injector<lv>::conditional;
		static constexpr bool rv_branch = injector<rv>::priority <= priority && injector<rv>::conditional;
		static constexpr bool conditional = lv_branch || rv_branch;

		template<class injection_t> using lv_branch_t = typename injector<lv>::template type<injection_t>;
		template<class injection_t> using rv_branch_t = typename injector<rv>::template type<injection_t>;

		template<class injection_t>
		using type = binary_expression<lv_branch_t<injection_t>, rv_branch_t<injection_t>, func>;
	};
	//OVERLOAD BLAS FUNCTIONS WE NEED THIS
	template<class lv, class rv, class ml>
	struct injector<binary_expression<lv, rv, BC::oper::dotproduct<ml>>, void> {
		static constexpr int priority = -1;
		static constexpr bool conditional = true;

		template<class injection_type> using type = injection_type;
	};

	template<class T>
	struct valid_injection_assignment {
		static constexpr bool conditional = false;
	};

	//checks for an apropriate assignment operator, to ensure that we are using the "injection" scheme
	template<class lv, class rv>
	struct valid_injection_assignment<BC::internal::binary_expression<lv, rv, BC::oper::assign>> {
		static constexpr bool conditional = true;
	};

	//checks for a valid assignment operation and if there is a BLAS function that is a valid injection
	template<class T>
	static constexpr bool INJECTION() {
		return valid_injection_assignment<std::decay_t<T>>::conditional && injector<std::decay_t<T>>::conditional;
	}

	//checks just for a BLAS method, if there is we need to utilize a substitution
	template<class T>
	static constexpr bool SUBSTITUTION() {
		return !valid_injection_assignment<std::decay_t<T>>::conditional && injector<std::decay_t<T>>::conditional;
	}


	template<class expression, class injection>
	using injection_t =  typename injector<std::decay_t<expression>>::template type<std::decay_t<injection>>;


}
}



#endif /* BLAS_INJECTION_EVALUATOR_H_ */
