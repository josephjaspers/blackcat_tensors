/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_EVALUATOR_H_
#define GPU_EVALUATOR_H_

#include "GPU_impl.cu"

namespace BC{
namespace gpu_impl {}
template<class core_lib>
class GPU_Evaluator {
    static int blocks(int sz) { return core_lib::blocks(sz); }
    static int threads() { return core_lib::threads(); }

public:
    template<int d>
    struct dimension {
        struct n1 { template<class T> static void eval(T to) { gpu_impl::eval<<<blocks(to.size()),threads()>>>(to);   }};
        struct n2 { template<class T> static void eval(T to) { gpu_impl::eval2d<<<blocks(to.size()),threads()>>>(to); }};
        struct n3 { template<class T> static void eval(T to) { gpu_impl::eval3d<<<blocks(to.size()),threads()>>>(to); }};
        struct n4 { template<class T> static void eval(T to) { gpu_impl::eval4d<<<blocks(to.size()),threads()>>>(to); }};
        struct n5 { template<class T> static void eval(T to) { gpu_impl::eval5d<<<blocks(to.size()),threads()>>>(to); }};
        using run = std::conditional_t<(d <= 1), n1,
                        std::conditional_t< d ==2, n2,
                            std::conditional_t< d == 3, n3,
                                std::conditional_t< d == 4, n4, n5>>>>;

        //These wonky specializations are essential for cuda to compile
        template<class T>
        static void eval(T to) {
            run::eval(to);
            cudaDeviceSynchronize();
        }

        template<template<class...> class T, class... Ts>
        static void eval(T<Ts...> to) {
            run::eval(to);
            cudaDeviceSynchronize();
        }
    };

    template<int d, class expr_t>
    static void nd_evaluator(expr_t expr) {
        dimension<d>::eval(expr);
    }



};
}




#endif /* GPU_EVALUATOR_H_ */
