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


		template<class, int, int, class...> class Matrix;
		template<class, int, class...> class Vector;


		//class variant of bool (good when you only want to use classes as templates)
		template<bool value>
		struct BOOL {
				static constexpr bool conditional = value;
		};

		template<class...>
		struct _list;

		template<class, class>  struct expression_substitution;

		template<class sub, template<class, class> class derived, class scalar_type, class ml>
		struct expression_substitution<sub, derived<scalar_type, ml>>{
				using type = derived<sub, ml>;
		};

//		//Vector Specialization
//		template<class, class>  struct expression_substitution;
//		template<class sub, template<class, int, class...> class derived, class scalar_type, class... modifiers, int rows>
//		struct expression_substitution<sub, derived<scalar_type, rows, modifiers...>>{
//				using type = derived<sub, rows, modifiers...>;
//		};
//		//Matrix Specialization
//		template<class, class>  struct expression_substitution;
//		template<class sub, template<class, int, int, class...> class derived, class scalar_type, class... modifiers, int rows, int cols>
//		struct expression_substitution<sub, derived<scalar_type, rows, cols, modifiers...>>{
//				using type = derived<sub, rows, cols, modifiers...>;
//		};

		template<class T>
		struct determine_scalar {
				using type = T;
		};
		template<template<class...> class tensor, class T, class... set>
		struct determine_scalar<tensor<T, set...>> {
				using type = typename determine_scalar<T>::type;
		};


		template<class T, class voider = void>
		struct determine_functor {
				using type = T;
		};


		template<class T>
		struct determine_functor<T, typename std::enable_if<MTF::isPrimitive<T>::conditional, void>::type> {
				using type = T*;
		};

		template<class tensor>
		struct determine_evaluation;

		template<bool, class T> struct if_AddRef { using type = T; };
		template<	   class T> struct if_AddRef<true, T> { using type = T&; };

		template<template<class, class> class tensor, class T, class ML>
		struct determine_evaluation<tensor<T, ML>> {

			using scalar_type = typename determine_scalar<T>::type;
			using type_ = tensor<scalar_type, ML>;
			using type  = typename if_AddRef<MTF::same<tensor<T, ML>, type_>::conditional, type_>::type;
		};

	}
}
#endif /* ADHOC_H_ */
