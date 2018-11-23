/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef EXPRESSION_TREE_FUNCTIONS_H_
#define EXPRESSION_TREE_FUNCTIONS_H_

#include "Tree_Evaluator_Common.h"

namespace BC{

class CPU;
class GPU;

namespace et     {
namespace tree {



template<class> struct scalar_modifer {
    enum mod {
        alpha = 0,
        beta = 0,
    };
};
template<> struct scalar_modifer<et::oper::add> {
    enum mod {
        alpha = 1,
        beta = 1,
    };
};
template<> struct scalar_modifer<et::oper::sub> {
    enum mod {
        alpha = -1,
        beta = 1
    };
};
template<> struct scalar_modifer<et::oper::add_assign> {
    enum mod {
        alpha = 1,
        beta = 1,
    };
};
template<> struct scalar_modifer<et::oper::sub_assign> {
    enum mod {
        alpha = -1,
        beta = 1,
    };
};
template<> struct scalar_modifer<et::oper::assign> {
    enum mod {
        alpha = 1,
        beta = 0,
    };
};


template<class T>
struct is_blas_func_impl { static constexpr bool value = false; };
template<> struct is_blas_func_impl<BC::CPU> { static constexpr bool value = true; };
template<> struct is_blas_func_impl<BC::GPU> { static constexpr bool value = true; };


template<class T> static constexpr bool is_blas_func() {
//    return is_blas_func_impl<T>::value;
    return std::is_base_of<BC::BLAS_FUNCTION, T>::value;
}

template<class T>
static constexpr bool is_linear_op() {
    return MTF::seq_contains<T, et::oper::add, et::oper::sub>;
}

template<class T>
static constexpr bool is_nonlinear_op() {
    return  !MTF::seq_contains<T, et::oper::add, et::oper::sub> && !is_blas_func<T>();
}
template<class T>
static constexpr int alpha_of() {
    return scalar_modifer<std::decay_t<T>>::mod::alpha;
}
template<class T>
static constexpr int beta_of() {
    return scalar_modifer<std::decay_t<T>>::mod::beta;
}


//trivial_blas_evaluation -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
template<class op, class core, int a, int b>//only apply update if right hand side branch
auto update_injection(injector<core,a,b> tensor) {
    static constexpr int alpha = a != 0 ? a * alpha_of<op>() : 1;
    static constexpr int beta = b != 0 ? b * beta_of<op>() : 1;
    return injector<core, alpha, beta>(tensor.data());
}


}
}
}




#endif /* EXPRESSION_TREE_FUNCTIONS_H_ */
