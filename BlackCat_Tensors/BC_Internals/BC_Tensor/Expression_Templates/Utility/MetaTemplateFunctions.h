/*
 * BC_MetaTemplate_Simple.h
 *
 *  Created on: Dec 12, 2017
 *      Author: joseph
 */

#ifndef SIMPLE_H_
#define SIMPLE_H_
#include <type_traits>
#include "../BlackCat_Internal_Definitions.h"
namespace BC {
namespace MTF {
	static constexpr int max(int x) { return x; }
	static constexpr int min(int x) { return x; }

	template<class... integers>
	static constexpr int max(int x, integers... ints) { return x > max (ints...) ? x : max(ints...); }

	template<class... integers>
	static constexpr int min(int x, integers... ints) { return x < min (ints...) ? x : min(ints...); }

		//short_hand for const cast
	template<class T> auto& cc(const T& var) { return const_cast<T&>(var); }

	template<class... Ts>
	struct sequence {
		template<class U> static constexpr bool of = false;
		template<class U> static constexpr bool contains = false;
		template<class U> static constexpr bool excludes = false;
	};
	template<class T> struct sequence<T> {
		template<class U> static constexpr bool of 		 = std::is_same<T, U>::value;
		template<class U> static constexpr bool contains = std::is_same<T, U>::value;
		template<class U> static constexpr bool excludes = !std::is_same<T,U>::value;
		using head = T;
		using tail = T;
	};

	template<class T, class... Ts> struct sequence<T, Ts...> {
		template<class U> static constexpr bool of 			= std::is_same<T, U>::value && sequence<Ts...>::template of<U>;
		template<class U> static constexpr bool contains 	= std::is_same<T, U>::value || sequence<Ts...>::template contains<U>;
		template<class U> static constexpr bool excludes 	= !std::is_same<T,U>::value && sequence<Ts...>::template excludes<U>;
		using head = T;
		using tail = typename sequence<Ts...>::tail;
	};

	template<class T, class... Ts> static constexpr bool seq_of = sequence<Ts...>::template of<T>;
	template<class T, class... Ts> static constexpr bool seq_contains = sequence<Ts...>::template contains<T>;
	template<class T, class... Ts> static constexpr bool seq_excludes = sequence<Ts...>::template excludes<T>;

	template<bool>
	struct constexpr_ternary_impl {
		template<class f1, class f2>
		static auto impl(f1 true_path, f2 false_path) {
			return true_path();
		}
	};

	template<>
	struct constexpr_ternary_impl<false> {
		template<class f1, class f2>
		static auto impl(f1 true_path, f2 false_path) {
			return false_path();
		}
	};

	template<bool cond, class f1, class f2>
	auto constexpr_ternary(f1 true_path, f2 false_path) {
		return constexpr_ternary_impl<cond>::impl(true_path, false_path);
	}

	template<bool>
	struct constexpr_if_impl {
		template<class f1>
		static auto impl(f1 path) {
			return path();
		}
	};
	template<>
	struct constexpr_if_impl<false> {
		template<class f1>
		static auto impl(f1 path) {
			return path();
		}
	};
	template<bool b,class f>
	auto constexpr_if(f path) {
		return constexpr_if_impl<b>::impl(path);
	}
}
}
#endif /* SIMPLE_H_ */
