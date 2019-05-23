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

#define BC_BIN_OP(...)\
	template<class Lv, class Rv>\
	BCINLINE \
	static auto apply (Lv&& lv, Rv&& rv) \
	-> decltype(__VA_ARGS__) {\
		return __VA_ARGS__;\
	}\
	template<class Lv, class Rv>\
	BCINLINE auto operator () (Lv&& lv, Rv&& rv) const \
	-> decltype(apply(lv, rv)) {\
		return apply(lv, rv);\
	}\
	template<class Lv, class Rv>\
	BCINLINE auto operator () (Lv&& lv, Rv&& rv) \
	-> decltype(apply(lv, rv)) {\
		return apply(lv, rv);\
	}



    struct scalar_mul {
    	BC_BIN_OP(lv * rv)
    };


    struct add : linear_operation, alpha_modifier<1> {
    	BC_BIN_OP(lv + rv)
    };

    struct mul {
    	BC_BIN_OP(lv * rv)
    };

    struct sub : linear_operation, alpha_modifier<-1> {
    	BC_BIN_OP(lv - rv)
    };

    struct div {
    	BC_BIN_OP(lv / rv)
    };

    struct assign : assignment_operation, beta_modifier<0>, alpha_modifier<1> {
    	BC_BIN_OP(lv = rv)
    };

    struct add_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
    	BC_BIN_OP(lv += rv)
    };

    struct mul_assign : assignment_operation {
    	BC_BIN_OP(lv *= rv)
    };

    struct sub_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
    	BC_BIN_OP(lv -= rv)
    };

    struct div_assign : assignment_operation {
    	BC_BIN_OP(lv /= rv)
    };



    struct equal {
    	BC_BIN_OP(lv == rv)
    };

    struct approx_equal {
    	static constexpr float epsilon = .01;
    	BC_BIN_OP(std::abs(lv - rv) < epsilon)
    };

    struct greater {
    	BC_BIN_OP(lv > rv)
    };

    struct lesser {
    	BC_BIN_OP(lv < rv)

    };

    struct greater_equal {
    	BC_BIN_OP(lv >= rv)
    };

    struct lesser_equal {
    	BC_BIN_OP(lv <= rv)
    };

    struct max {
    	BC_BIN_OP(lv > rv ? lv : rv)
    };

    struct min {
    	BC_BIN_OP(lv < rv ? lv : rv)
    };

    struct AND {
    	BC_BIN_OP(lv && rv)
    };

    struct OR {
    	BC_BIN_OP(lv || rv)
    };

    struct XOR {
    	BC_BIN_OP(lv ^ rv)
    };
}
}

#undef BC_BIN_OP
#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

