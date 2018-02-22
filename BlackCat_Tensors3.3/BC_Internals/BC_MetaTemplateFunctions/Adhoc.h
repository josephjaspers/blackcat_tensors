/*
 * BC_MetaTemplate_EssentialMethods.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef ADHOC_H_
#define ADHOC_H_

#include "Simple.h"
#include <type_traits>

namespace BC {
	namespace MTF {

		//EXPRESSION_TYPE----------------------------------------------------------------------------------------------
		template<class, class>  struct expression_substitution;

		template<class sub, template<class, class> class derived, class scalar_type, class ml>
		struct expression_substitution<sub, derived<scalar_type, ml>>{
				using type = derived<sub, ml>;
		};
		//SCALAR_TYPE----------------------------------------------------------------------------------------------
		template<class T>
		struct determine_scalar {
				using type = T;
		};
		template<template<class...> class tensor, class T, class... set>
		struct determine_scalar<tensor<T, set...>> {
				using type = typename determine_scalar<T>::type;
		};
		//EVALUTION_TYPE----------------------------------------------------------------------------------------------
		template<class tensor>
		struct determine_evaluation;

		template<bool, class T> struct if_AddRef { using type = T; };
		template<	   class T> struct if_AddRef<true, T> { using type = T&; };
		template<class T> struct ref { using type = T&; };

		template<template<class, class> class tensor, class T, class ML>
		struct determine_evaluation<tensor<T, ML>> {

			using type = typename MTF::IF_ELSE<isPrimitive<T>::conditional,
									typename ref<tensor<T, ML>>::type,
									tensor<typename determine_scalar<T>::type, ML>>::type;
		};
	}
}
#endif /* ADHOC_H_ */
