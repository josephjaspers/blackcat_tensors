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

namespace BC {
namespace et {
namespace tree {

//default
template<class> struct scalar_modifer {
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

template<class T> static constexpr bool is_blas_func() {
    return std::is_base_of<BC::BLAS_FUNCTION, T>::value;
}

template<class T>
static constexpr bool is_linear_op() {
    return MTF::seq_contains<T, et::oper::add, et::oper::sub>;
}

template<class T>
static constexpr bool is_linear_assignment_op() {
    return MTF::seq_contains<T, et::oper::add_assign, et::oper::sub_assign>;
}



template<class T>
static constexpr bool is_nonlinear_op() {
    return  !is_linear_op<T>() && !is_blas_func<T>();
}
template<class T>
static constexpr int alpha_of() {
    return scalar_modifer<std::decay_t<T>>::mod::alpha;
}
template<class T>
static constexpr int beta_of() {
    return scalar_modifer<std::decay_t<T>>::mod::beta;
}


//entirely_blas_expr -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
template<class op, bool prior_eval, class core, int a, int b>//only apply update if right hand side branch
auto update_injection(injector<core,a,b> tensor) {
    static constexpr int alpha =  a * alpha_of<op>();
    static constexpr int beta  = prior_eval ? 1 : 0;
    return injector<core, alpha, beta>(tensor.data());
}


}
}
}




#endif /* EXPRESSION_TREE_FUNCTIONS_H_ */
