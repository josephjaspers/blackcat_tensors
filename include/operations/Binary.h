/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_

#include <type_traits>
#include <cmath>

#include "Tags.h"

namespace BC {
namespace oper {


    struct scalar_mul {
        template<class lv, class rv>
        BCINLINE auto operator ()(lv l, rv r) const {
            return l * r;
        }
    };


    struct add : linear_operation, alpha_modifier<1> {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l + r;
        }
    };

    struct mul {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l * r;
        }
    };

    struct sub : linear_operation, alpha_modifier<-1> {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l - r;
        }
    };

    struct div {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l / r;
        }
    };

    struct assign : assignment_operation, beta_modifier<0>, alpha_modifier<1> {
        template<class lv, class rv>
		BCINLINE auto& operator ()(lv& l, rv r) const {
            return meta::bc_const_cast(l) = r;
        }
    };

    struct add_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv& l, rv r) const {
            return meta::bc_const_cast(l) += r;
        }
    };

    struct mul_assign : assignment_operation {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv& l, rv r) const {
            return meta::bc_const_cast(l) *= r;
        }
    };

    struct sub_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv& l, rv r) const {
            return meta::bc_const_cast(l) -= r;
        }
    };

    struct div_assign : assignment_operation {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv& l, rv r) const {
            return meta::bc_const_cast(l) /= r;
        }
    };

    struct equal {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l == r;
        }
    };

    struct approx_equal {

    	static constexpr float epsilon = .01;

        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return std::abs(l - r) < epsilon;
        }
    };

    struct greater {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l > r;
        }
    };

    struct lesser {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l < r;
        }
    };

    struct greater_equal {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l >= r;
        }
    };

    struct lesser_equal {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l <= r;
        }
    };

    struct max {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l > r ? l : r;
        }
    };

    struct min {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l < r ? l : r;
        }
    };

    struct AND {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l && r;
        }
    };

    struct OR {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l || r;
        }
    };

    struct XOR {
        template<class lv, class rv>
		BCINLINE auto operator ()(lv l, rv r) const {
            return l ^ r;
        }
    };
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

