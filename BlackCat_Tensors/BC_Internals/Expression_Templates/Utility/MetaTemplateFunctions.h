/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef SIMPLE_H_
#define SIMPLE_H_
#include <type_traits>
#include "../Internal_Common.h"
namespace BC {
namespace MTF {

    template<class t,class u> static constexpr bool is_same = std::is_same<t,u>::value;

    __BCinline__ static constexpr int max(int x) { return x; }
    __BCinline__ static constexpr int min(int x) { return x; }

    template<class... integers>
    __BCinline__ static constexpr int max(int x, integers... ints) { return x > max (ints...) ? x : max(ints...); }

    template<class... integers>
    __BCinline__ static constexpr int min(int x, integers... ints) { return x < min (ints...) ? x : min(ints...); }

    static constexpr int sum(int x) { return x; }

    template<class... integers>
    static constexpr int sum(int x, integers... ints) { return x + sum(ints...); }

        //short_hand for const cast
    template<class T> auto& cc(const T& var) { return const_cast<T&>(var); }

    template<class T, class... Ts> using head_t = T;

    template<class T, class... Ts>
    struct seq_contains_impl {
        static constexpr bool value= false;
    };
    template<class T, class U, class... Ts>
        struct seq_contains_impl<T,U,Ts...> {
            static constexpr bool value= std::is_same<T,U>::value || seq_contains_impl<T,Ts...>::value;
        };

    template<class T, class... Ts>
    struct seq_of_impl {
        static constexpr bool value = false;
    };
    template<class T, class U>
    struct seq_of_impl<T,U> {
        static constexpr bool value = std::is_same<T,U>::value;
    };
    template<class T, class U, class... Ts>
    struct seq_of_impl<T,U,Ts...> {
        static constexpr bool value = std::is_same<T,U>::value || seq_of_impl<T,Ts...>::value;
    };

    template<class T, class U, class... Ts> static constexpr bool seq_of = seq_of_impl<T,U,Ts...>::value;

    template<class... Ts> static constexpr bool seq_contains = seq_contains_impl<Ts...>::value;


    template<int index, class... Ts>
    struct get_impl;

    template<int index>
    struct get_impl<index>{
        template<class T, class... Ts>
        static auto impl(T head, Ts... params) {
             return get_impl<index - 1>::impl(params...);
        }
    };
    template<>
    struct get_impl<0>{
        template<class T, class... Ts>
        static auto impl(T head, Ts... params) {
             return head;
        }
    };
    template<int index, class... Ts>
    auto get(Ts... params) {
        return get_impl<index>::impl(params...);
    }


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


    //---------------------

}
}
#endif /* SIMPLE_H_ */
