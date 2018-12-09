/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_

#include "../Internal_Common.h"

namespace BC {
namespace et     {
namespace oper {
template<class T> struct rm_const { using type = T; };
template<class T> struct rm_const<const T&> { using type = T&; };

/*
 * 0 = a Tensor (base case)
 * 1 = function
 * 2 = Multiplication/division
 * 3 = addition/subtraction
 * 4 = assignments
 *
 */
class assignment {};
class matrix_oper {};

    struct scalar_mul {
        //this is just a flag for gemm, it is the same as multiplication though
        template<class lv, class rv> __BCinline__ auto operator ()(lv l, rv r) const {
            return l * r;
        }
    };


    struct add : matrix_oper {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l + r;
        }
    };

    struct mul {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l * r;
        }
    };

    struct sub : matrix_oper {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l - r;
        }
    };

    struct div {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l / r;
        }
    };
    struct combine {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l;
        }
    };
    struct assign : matrix_oper {
        template<class lv, class rv> __BCinline__ auto& operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) = r);
        }
    };

    struct add_assign : matrix_oper, assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) += r);
        }
    };

    struct mul_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) *= r);
        }
    };

    struct sub_assign : matrix_oper, assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) -= r);
        }
    };

    struct div_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) /= r);
        }
    };
    struct alias_assign : assignment {
        template<class lv, class rv> __BCinline__ auto& operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) = r);
        }
    };

    struct alias_add_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) += r);
        }
    };

    struct alias_mul_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) *= r);
        }
    };

    struct alias_sub_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) -= r);
        }
    };

    struct alias_div_assign : assignment {
        template<class lv, class rv> __BCinline__  auto operator ()(lv& l, rv r) const {
            return (const_cast<typename rm_const<lv&>::type>(l) /= r);
        }
    };
    struct equal {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l == r;
        }
    };
    struct approx_equal {

    	static constexpr float epsilon = .0001;

        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return abs(l - r) < epsilon;
        }
    };


    struct greater {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l > r;
        }
    };
    struct lesser {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l < r;
        }
    };
    struct greater_equal {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l >= r;
        }
    };
    struct lesser_equal {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l <= r;
        }
    };
    struct max {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l > r ? l : r;
        }
    };
    struct min {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l < r ? l : r;
        }
    };
    struct AND {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l && r;
        }
    };
    struct OR {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l || r;
        }
    };
    struct XOR {
        template<class lv, class rv> __BCinline__  auto operator ()(lv l, rv r) const {
            return l ^ r;
        }
    };
}
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

