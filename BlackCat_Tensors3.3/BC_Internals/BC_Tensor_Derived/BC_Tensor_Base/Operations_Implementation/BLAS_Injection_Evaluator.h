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

class GPU;
class CPU;

namespace function {

//priority -1 == non-rotateable (injection not legal)
//priority 0 == low
//priority 1 == high
/*
 * a low priority may be after a high priority (ensure priority order)
 * but a low priority preceding a high-priority operation will result in non-injectable segment
 */
template<class> struct dotproduct;
template<class> struct PRIORITY { static constexpr int value = -1;};
template<> struct PRIORITY<add> { static constexpr int value = 0; };
template<> struct PRIORITY<sub> { static constexpr int value = 0; };
template<> struct PRIORITY<mul> { static constexpr int value = 1; };
template<> struct PRIORITY<div> { static constexpr int value = 1; };
template<> struct PRIORITY<add_assign> { static constexpr int value = 0; };
//template<> struct PRIORITY<sub_assign> { static constexpr int value = 0; }; //this will eventually be allowed but for now NO
template<> struct PRIORITY<assign> { static constexpr int value = 1; };

template<class T> struct assignment_conversion {
	using type = T;
	static constexpr int scalar = 0; //beta scalar (no conversion = clear)
};
//add assign / sub_assigns get downcasted
template<> struct assignment_conversion<add_assign> {
	using type = assign;
	static constexpr int scalar = 1; //beta scalar for BLAS

};
//Theoretically we can support this but its a mindfudge, so will add later
//template<> struct assignment_conversion<sub_assign> {
//	using type = assign;
//	static constexpr int beta_scalar = 1;//beta scalar for BLAS
//	static constexpr int alpha_scalar_multipler = -1;
//
//};



//=unary_expressions have a priority of 1

}

namespace internal {
	template<class> struct Tensor_Core;

	template<class T> static constexpr bool INJECTION();
	//shorthands
	template<class lv, class rv, class oper> using be = binary_expression<lv, rv, oper>;
	template<class v, class oper> using ue = unary_expression<v, oper>;

	template<class T> using enable_if_BLAS = std::enable_if_t<std::is_base_of<BC::BLAS_FUNCTION, T>::value>;
	template<class T> using enable_if_core = std::enable_if_t<std::is_base_of<BC_Core, T>::value>;


	//DEFAULT -- (non-BLAS injection on unknown/non-specified functions)
	template<class T, class enabler = void> struct injector {
		static constexpr bool conditional = false;
		static constexpr int priority = - 1;
		template<class injection_type>
		using type = T;
	};
//	//DETECT BLAS FUNCTION ??? doesn't work, currently requires specialization for all BLAS functions
//	//This actually isn't so bad, but would make the framework a bit better (if it did work)
//	template<class lv, class rv, class func>
//	struct injector<be<lv, rv, func>, enable_if_BLAS<func>> {
//		static constexpr bool conditional = true;
//		using type = Tensor_Core<Matrix<float, BC::CPU>>;
//	};
	//ALL UNARY FUNCTIONS HAVE HIGH_PRIORITY
	template<class V, class func>
	struct injector<ue<V, func>> {
		static constexpr int priority = 1;
		static constexpr bool conditional = injector<V>::priority <= priority && injector<V>::conditional;

		template<class injection_type>
		using type = std::conditional_t<conditional, ue<typename injector<V>::template type<injection_type>, func>, ue<V, func>>;
	};
	//BINARY EXPRESSION - detect precedence of the func (operand),
	template<class lv, class rv, class func>
	struct injector<be<lv, rv, func>> {
		using self = injector<be<lv,rv, func>>;
		static constexpr int priority = function::PRIORITY<func>::value;
		static constexpr bool lv_branch = injector<lv>::priority <= priority && injector<lv>::conditional;
		static constexpr bool rv_branch = injector<rv>::priority <= priority && injector<rv>::conditional;
		static constexpr bool conditional = lv_branch || rv_branch;

		template<class injection_type>
		using type =
				std::conditional_t<lv_branch, 													//if
					be<typename injector<lv>::template type<injection_type>, rv, func>,	//then

						std::conditional_t<rv_branch,							//else if
							be<lv, typename injector<rv>::template type<injection_type>, func>,//then
							self													//else
						>>;

	};

	template<class lv, class rv, class ml>
	struct injector<be<lv, rv, BC::function::dotproduct<ml>>, void> {
		static constexpr int priority = -1;
		static constexpr bool conditional = true;

		//injection_type is the "core" injection (slice/chunk/row/core) this causes a "drop-in" replacement for the blas-function
		template<class injection_type> using type = std::decay_t<injection_type>;
	};


	template<class T>
	static constexpr bool INJECTION() {
		return injector<std::decay_t<T>>::conditional;
	}

	template<class expression, class injection>
	using injection_t =  typename injector<std::decay_t<expression>>::template type<std::decay_t<injection>>;


}
}



#endif /* BLAS_INJECTION_EVALUATOR_H_ */
