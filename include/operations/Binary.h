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

#define BC_FORWARD_TO_APPLY\
		template<class Lv, class Rv>													\
		BCINLINE auto operator () (Lv&& lv, Rv&& rv) const 								\
		-> decltype(apply(lv, rv)) {													\
			return apply(lv, rv);														\
		}																				\
																			\

#define BC_FORWARD_DEF(...)															\
	template<class Lv, class Rv>													\
	BCINLINE 																		\
	static auto apply (Lv&& lv, Rv&& rv) 											\
	-> decltype(__VA_ARGS__) {														\
		return __VA_ARGS__;															\
	}																				\
	BC_FORWARD_TO_APPLY

#define BC_ADVANCED_FORWARD_DEF(...)												\
	template<class Lv, class Rv>													\
	BCINLINE 																		\
	static Lv&& apply (Lv&& lv, Rv&& rv) {											\
		__VA_ARGS__;																\
	}																				\
	BC_FORWARD_TO_APPLY


#define BC_BACKWARD_LV_DEF(...)						\
	template<class Delta, class Lv, class Rv>				\
	BCINLINE static auto lv_dx(Lv&& lv, Rv&& rv, Delta&& dy) 	\
	{												\
		return __VA_ARGS__;								\
	}												\

#define BC_BACKWARD_RV_DEF(...)						\
	template<class Delta, class Lv, class Rv>				\
	BCINLINE static auto rv_dx(Lv&& lv, Rv&& rv, Delta&& dy) 	\
	{												\
		return __VA_ARGS__;								\
	}												\

    struct Scalar_Mul {
    	BC_FORWARD_DEF(lv * rv)
    	BC_BACKWARD_LV_DEF(rv)
    	BC_BACKWARD_RV_DEF(lv)
    } scalar_mul;


    struct Add : linear_operation, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv + rv)
    	BC_BACKWARD_LV_DEF(1)
    	BC_BACKWARD_RV_DEF(1)
    } add;

    struct Mul {
    	BC_FORWARD_DEF(lv * rv)
    	BC_BACKWARD_LV_DEF(rv)
    	BC_BACKWARD_RV_DEF(lv)
    } mul;

    struct Sub : linear_operation, alpha_modifier<-1> {
    	BC_FORWARD_DEF(lv - rv)
    	BC_BACKWARD_LV_DEF(1)
    	BC_BACKWARD_RV_DEF(-1)
    } sub;

    struct Div {
    	BC_FORWARD_DEF(lv / rv)
    	BC_BACKWARD_LV_DEF(1/rv)
    	BC_BACKWARD_RV_DEF(lv)
    } div;

    struct Assign : assignment_operation, beta_modifier<0>, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv = rv)
    } assign;

    struct Add_Assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
    	BC_FORWARD_DEF(lv += rv)
    } add_assign;

    struct Mul_Assign : assignment_operation {
    	BC_FORWARD_DEF(lv *= rv)
    } mul_assign;

    struct Sub_Assign : linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
    	BC_FORWARD_DEF(lv -= rv)
    } sub_assign;

    struct Div_Assign : assignment_operation {
    	BC_FORWARD_DEF(lv /= rv)
    } div_assign;



    struct Equal {
    	BC_FORWARD_DEF(lv == rv)
    } equal;

    struct Approx_Equal {
    	static constexpr float epsilon = .01;
    	BC_FORWARD_DEF(std::abs(lv - rv) < epsilon)
    } approx_equal;

    struct Greater {
    	BC_FORWARD_DEF(lv > rv)
    } greater;

    struct Lesser {
    	BC_FORWARD_DEF(lv < rv)
    } lesser;

    struct Greater_Equal {
    	BC_FORWARD_DEF(lv >= rv)
    } greater_equal;

    struct Lesser_Equal {
    	BC_FORWARD_DEF(lv <= rv)
    } lesser_equal;

    struct Max {
    	BC_FORWARD_DEF(lv > rv ? lv : rv)
    } max;

    struct Min {
    	BC_FORWARD_DEF(lv < rv ? lv : rv)
    } min;

    struct And {
    	BC_FORWARD_DEF(lv && rv)
    } and_;

    struct Or {
    	BC_FORWARD_DEF(lv || rv)
    } or_;

    struct Xor {
    	BC_FORWARD_DEF(lv ^ rv)
    } xor_;


#ifdef __CUDACC__
#define IF_DEVICE_MODE(...) __VA_ARGS__
#define IF_HOST_MODE(...)
#else
#define IF_DEVICE_MODE(...)
#define IF_HOST_MODE(...) __VA_ARGS__
#endif


struct Atomic_Add:
	linear_assignment_operation, beta_modifier<1>, alpha_modifier<1> {
	BC_ADVANCED_FORWARD_DEF(
		IF_HOST_MODE(
			BC_omp_atomic__
			lv += rv;
			return lv;
		)
		IF_DEVICE_MODE(
			atomicAdd(&lv, rv);
			return lv;
		)
	)
} atomic_add;


struct Atomic_Mul:
		assignment_operation {
		BC_ADVANCED_FORWARD_DEF(
			IF_HOST_MODE(
				BC_omp_atomic__
				lv *= rv;
				return lv;
			)
			IF_DEVICE_MODE(
				static_assert(
					std::is_same<void, Lv>::value,
					"BLACKCAT_TENSORS: Atomic-reduction mul-assign is currently not available on the GPU");

			)
	)
} atomic_mul;

struct Atomic_Sub:
	linear_assignment_operation, beta_modifier<1>, alpha_modifier<-1> {
		BC_ADVANCED_FORWARD_DEF(
			IF_HOST_MODE(
				BC_omp_atomic__
				lv -= rv;
				return lv;
			)
			IF_DEVICE_MODE(
				atomicAdd(&lv, -rv);
				return lv;
			)
		)
} atomic_sub;


struct Atomic_Div: assignment_operation {
	BC_ADVANCED_FORWARD_DEF(
		IF_HOST_MODE(
			BC_omp_atomic__
			lv /= rv;
			return lv;
		)

		IF_DEVICE_MODE(
			static_assert(
				std::is_same<void, Lv>::value,
				"BLACKCAT_TENSORS: Atomic-reduction div-assign is currently not available on the GPU");

		)
	)
} atomic_div;

}
}

#undef BC_FORWARD_DEF
#undef BC_BACKWARD_DEF
#undef BC_FORWARD_TO_APPLY

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

