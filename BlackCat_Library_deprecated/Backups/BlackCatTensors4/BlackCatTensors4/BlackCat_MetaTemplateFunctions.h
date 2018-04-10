/*
 * BlackCat_MetaTemplateFunctions.h
 *
 *  Created on: Nov 29, 2017
 *      Author: joseph
 */

#ifndef BLACKCAT_METATEMPLATEFUNCTIONS_H_
#define BLACKCAT_METATEMPLATEFUNCTIONS_H_

namespace BC_MTF {

	template<class ... list>
	struct t_list;

	template<int a, int b>
	struct max {
		static constexpr int value = a > b ? a : b;
		static constexpr bool isA = a > b;
		static constexpr bool isB = a < b;

	};
	template<int a, int b>
	struct min {
		static constexpr int value = a > b ? b : a;
		static constexpr bool isA = a < b;
		static constexpr bool isB = b < a;

	};
	template<int a, int b>
	struct equal {
		static constexpr bool conditional = a == b;
	};

	template<class ...>
	struct isTrue {
		static constexpr bool conditional = true;
	};
	template<>
	struct isTrue<> {
		static constexpr bool conditional = true;
	};
	template<class ...>
	struct isFalse {
		static constexpr bool conditional = true;
	};

	template<>
	struct isFalse<> {
		static constexpr bool conditional = true;
	};

	template<int getIndex, int currIndex, class ... list>
	struct indexer;

	template<int getIndex, int currIndex, class currEle, class ... list>
	struct indexer<getIndex, currIndex, currEle, list...> {
		using type = typename indexer<getIndex + 1, currIndex, list...>::type;
	};
	template<int id, class currEle, class ... list>
	struct indexer<id, id, currEle, list...> {
		using type = currEle;
	};

	//------------------------------------------------------RemoveHead------------------------------------------------------//
	template<template<class > class conditional, class ... list>
	struct remove_head_impl;

	template<bool booleaner, class front, class ... list>
	struct checkBool_head {
		using type = t_list<front, list...>;
	};
	template<class front, class ... list>
	struct checkBool_head<false, front, list...> {
		using type = t_list<list...>;
	};

	template<template<class > class conditional_type, class front, class ... list>
	struct remove_head_impl<conditional_type, front, list...> {
		using type = typename checkBool_head<conditional_type<front>::conditional, front, list...>::type;
	};

	//------------------------------------------------------Flip------------------------------------------------------//

	template<class flipped, class ... list>
	struct invert;

	template<class ... flipped_list, class front, class ... list>
	struct invert<t_list<flipped_list...>, front, list...> {
		using type = typename invert<t_list<front, flipped_list...>, list...>::type;
	};

	template<class ... flipped_list, class front>
	struct invert<t_list<flipped_list...>, front> {
		using type = invert<t_list<front, flipped_list...>>;
	};

	//---------------------------------------------------------------------------------------------------------------//
	template<class extract_from, template<class ...> class extract_to, class extract_to_left_params = void, class extract_to_right_params = void>
	struct extract;

	template<template<class ...> class extract_from, class... data, template<class...> class extract_to>
	struct extract<extract_from<data...>, extract_to, void, void> {
		using type = extract_to<data...>;
	};
	template<template<class ...> class extract_from, class... data, template<class...> class extract_to, class left, class right>
	struct extract<extract_from<data...>, extract_to, left, right> {
		using type = extract_to<left, data..., right>;
	};

	template<template<class ...> class extract_from, class... data, template<class...> class extract_to, template<class...> class left, class... left_params, template<class...> class right, class... right_params>
	struct extract<extract_from<data...>, extract_to, left<left_params...>, right<right_params...>> {
		using type = extract_to<left_params..., data..., right_params...>;
	};

	//---------------------------------------------------------------------------------------------------------------//
	template<class f, class ... list>
	struct t_list<f, list...> {
		using first = f;
		using last = typename t_list<list...>::last;

		template<int i>
		struct index_of {
			using type = typename indexer<i, 0, list...>::type;
		};

		template<template<class > class conditional>
		struct remove_head_if {
			using type = typename remove_head_impl<conditional, f, list...>::type;
		};

		template<template<class > class conditional>
		struct recursive_remove_head_if {
			using type = typename remove_head_impl<conditional, f, list...>::type;
		};

		struct remove_head {
			using type = typename remove_head_impl<isTrue, f, list...>::type;
		};

		struct flip {
			using type = typename invert<f, list...>::type;
		};
	};

	template<class f>
	struct t_list<f> {
		using first = f;
		using last = f;

		template<int i>
		struct index_of {
			using type = typename indexer<i, 0, f>::type;
		};

		template<template<class > class cond>
		struct remove_head_if {
			using type = typename checkBool_head<cond<f>::conditional, void, f>::type;
		};

		template<template<class > class cond>
		struct recursively_remove_head_if {
			using type = typename checkBool_head<cond<f>::conditional, void, f>::type;
		};

		struct remove_head {
			using type = f;
		};

		struct flip {
			using type = f;
		};
	};
}

#endif /* BLACKCAT_METATEMPLATEFUNCTIONS_H_ */
