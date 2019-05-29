/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_BINARY_FUNCTORS_H_
#define EXPRESSION_BINARY_FUNCTORS_H_


#include "Operation_Traits.h"
#include "Tags.h"

#include <type_traits>
#include <cmath>


namespace BC {
namespace oper {


#define BC_FORWARD_DEF(...)															\
	template<class Lv, class Rv>													\
	BCINLINE 																		\
	static auto apply (Lv&& lv, Rv&& rv) 											\
	-> decltype(__VA_ARGS__) {														\
		return __VA_ARGS__;															\
	}																				\
																					\
	template<class Lv, class Rv>													\
	BCINLINE 																		\
	static auto forward_propagate (Lv&& lv, Rv&& rv) 								\
	-> decltype(apply(operation_traits<Lv>::select_on_forward_propagate(lv), 		\
						operation_traits<Rv>::select_on_forward_propagate(rv))) 	\
	{																				\
		return apply(operation_traits<Lv>::select_on_forward_propagate(lv), 		\
						operation_traits<Rv>::select_on_forward_propagate(rv));		\
	}																				\
																					\
	template<class Lv, class Rv>													\
	BCINLINE auto operator () (Lv&& lv, Rv&& rv) const 								\
	-> decltype(apply(lv, rv)) {													\
		return apply(lv, rv);														\
	}																				\


#define BC_BACKWARD_DEF(...)														\
	template<class Delta, class Lv_x, class Rv_x, class Lv, class Rv>				\
	BCINLINE 																		\
	static void backward_propagate(Delta&& dy, 										\
									Lv_x&& lv_x, 									\
									Rv_x&& rv_x, 									\
									Lv&& lv_, 										\
									Rv&& rv_) 										\
	{																				\
			auto&& lv = operation_traits<Lv>::select_on_backward_propagate(lv_);	\
			auto&& rv = operation_traits<Rv>::select_on_backward_propagate(rv_);	\
			__VA_ARGS__;															\
	}																				\



    struct scalar_mul {
    	BC_FORWARD_DEF(lv * rv)
    	BC_BACKWARD_DEF(lv.backward(dy * rv_x), rv.backward(lv_x * dy));
    };


    struct add : linear_operation, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv + rv)
    	BC_BACKWARD_DEF(lv.backward(dy), rv.backward(dy));

    };

    struct mul {
    	BC_FORWARD_DEF(lv * rv)
    	BC_BACKWARD_DEF(lv.backward(dy * rv_x), rv.backward(lv_x * dy))
    };

    struct sub : linear_operation, alpha_modifier<-1> {
    	BC_FORWARD_DEF(lv - rv)
    	BC_BACKWARD_DEF(lv.backward(dy), rv.backward(-dy))
    };

    struct div {
    	BC_FORWARD_DEF(lv / rv)
    };

    struct assign : assignment_operation, beta_modifier<0>, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv = rv)
    };

    struct add_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv += rv)
    };

    struct mul_assign : assignment_operation {
    	BC_FORWARD_DEF(lv *= rv)
    };

    struct sub_assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
    	BC_FORWARD_DEF(lv -= rv)
    };

    struct div_assign : assignment_operation {
    	BC_FORWARD_DEF(lv /= rv)
    };



    struct equal {
    	BC_FORWARD_DEF(lv == rv)
    };

    struct approx_equal {
    	static constexpr float epsilon = .01;
    	BC_FORWARD_DEF(std::abs(lv - rv) < epsilon)
    };

    struct greater {
    	BC_FORWARD_DEF(lv > rv)
    };

    struct lesser {
    	BC_FORWARD_DEF(lv < rv)

    };

    struct greater_equal {
    	BC_FORWARD_DEF(lv >= rv)
    };

    struct lesser_equal {
    	BC_FORWARD_DEF(lv <= rv)
    };

    struct max {
    	BC_FORWARD_DEF(lv > rv ? lv : rv)
    };

    struct min {
    	BC_FORWARD_DEF(lv < rv ? lv : rv)
    };

    struct AND {
    	BC_FORWARD_DEF(lv && rv)
    };

    struct OR {
    	BC_FORWARD_DEF(lv || rv)
    };

    struct XOR {
    	BC_FORWARD_DEF(lv ^ rv)
    };
}
}

#undef BC_FORWARD_DEF
#undef BC_BACKWARD_DEF
#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

