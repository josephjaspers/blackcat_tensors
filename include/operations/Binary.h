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
		template<class Lv, class Rv>							\
		BCINLINE auto operator () (Lv&& lv, Rv&& rv) const 		\
		-> decltype(apply(lv, rv)) {							\
			return apply(lv, rv);								\
		}														\
																\

#define BC_FORWARD_DEF(...)						\
	template<class Lv, class Rv>				\
	BCINLINE 									\
	static auto apply (Lv&& lv, Rv&& rv) 		\
	-> decltype(__VA_ARGS__) {					\
		return __VA_ARGS__;						\
	}											\
	BC_FORWARD_TO_APPLY

#define BC_ADVANCED_FORWARD_DEF(...)			\
	template<class Lv, class Rv>				\
	BCINLINE 									\
	static Lv&& apply (Lv&& lv, Rv&& rv) {		\
		__VA_ARGS__;							\
	}											\
	BC_FORWARD_TO_APPLY

    struct Scalar_Mul {
    	BC_FORWARD_DEF(lv * rv)
    } scalar_mul;


    struct Add : linear_operation {
    	using alpha_modifier = BC::traits::Integer<1>;
    	BC_FORWARD_DEF(lv + rv)
    } add;

    struct Mul {
    	BC_FORWARD_DEF(lv * rv)

    } mul;

    struct Sub : linear_operation {
    	using alpha_modifier = BC::traits::Integer<-1>;
    	BC_FORWARD_DEF(lv - rv)
    } sub;

    struct Div {
    	BC_FORWARD_DEF(lv / rv)
    } div;

    struct Fuse {
    	BC_FORWARD_DEF(lv)
    } fuse;


    struct Assign : assignment_operation {
    	using alpha_modifier = BC::traits::Integer<1>;
    	using beta_modifier = BC::traits::Integer<0>;
    	BC_FORWARD_DEF(lv = rv)
    } assign;

    struct Add_Assign : linear_assignment_operation {
    	using alpha_modifier = BC::traits::Integer<1>;
    	using beta_modifier = BC::traits::Integer<1>;
    	BC_FORWARD_DEF(lv += rv)
    } add_assign;

    struct Mul_Assign : assignment_operation {
    	using alpha_modifier = BC::traits::Integer<1>;
    	using beta_modifier = BC::traits::Integer<1>;
    	BC_FORWARD_DEF(lv *= rv)
    } mul_assign;

    struct Sub_Assign : linear_assignment_operation {
    	using alpha_modifier = BC::traits::Integer<-1>;
    	using beta_modifier = BC::traits::Integer<1>;
    	BC_FORWARD_DEF(lv -= rv)
    } sub_assign;

    struct Div_Assign : assignment_operation {
    	using alpha_modifier = BC::traits::Integer<1>;
    	using beta_modifier = BC::traits::Integer<1>;
    	BC_FORWARD_DEF(lv /= rv)
    } div_assign;



    struct Equal {
    	BC_FORWARD_DEF(lv == rv)
    } equal;

    struct Approx_Equal {
    	BC_FORWARD_DEF(std::abs(lv - rv) < .01)
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


    struct Make_Pair {
    	BC_FORWARD_DEF(BC::traits::make_pair(lv, rv))
    } make_pair;

#ifdef __CUDACC__
#define IF_DEVICE_MODE(...) __VA_ARGS__
#define IF_HOST_MODE(...)
#else
#define IF_DEVICE_MODE(...)
#define IF_HOST_MODE(...) __VA_ARGS__
#endif


struct Host_Atomic_Add:
	Add_Assign {
	BC_ADVANCED_FORWARD_DEF(
			BC_omp_atomic__
			lv += rv;
			return lv;
	)
} host_atomic_add;


struct Host_Atomic_Mul:
		Mul_Assign {
		BC_ADVANCED_FORWARD_DEF(
				BC_omp_atomic__
				lv *= rv;
				return lv;
	)
} host_atomic_mul;

struct Host_Atomic_Sub:
	Sub_Assign {
		BC_ADVANCED_FORWARD_DEF(
				BC_omp_atomic__
				lv -= rv;
				return lv;
		)
} host_atomic_sub;


struct Host_Atomic_Div: Div_Assign {
	BC_ADVANCED_FORWARD_DEF(
			BC_omp_atomic__
			lv /= rv;
			return lv;
	)
} host_atomic_div;

//----------------------------------------
namespace detail {
template<class T>
static constexpr bool is_host = std::is_same<T, host_tag>::value;
}


#ifdef __CUDACC__
struct Device_Atomic_Add: Add_Assign {
	BC_ADVANCED_FORWARD_DEF(
			atomicAdd(&lv, rv);
			return lv;
	)
} device_atomic_add;


struct Device_Atomic_Mul: Mul_Assign {
		BC_ADVANCED_FORWARD_DEF(
				static_assert(
					std::is_same<void, Lv>::value,
					"BLACKCAT_TENSORS: Atomic-reduction mul-assign is currently not available on the GPU");
	)
} device_atomic_mul;

struct Device_Atomic_Sub: Sub_Assign{
		BC_ADVANCED_FORWARD_DEF(
				atomicAdd(&lv, -rv);
				return lv;
		)
} device_atomic_sub;


struct Device_Atomic_Div: Div_Assign {
	BC_ADVANCED_FORWARD_DEF(
			static_assert(
				std::is_same<void, Lv>::value,
				"BLACKCAT_TENSORS: Atomic-reduction div-assign is currently not available on the GPU");
	)
} device_atomic_div;




template<class SystemTag>
using Atomic_Add = std::conditional_t<detail::is_host<SystemTag>,
		Host_Atomic_Add, Device_Atomic_Add>;
template<class SystemTag>
using Atomic_Sub = std::conditional_t<detail::is_host<SystemTag>,
		Host_Atomic_Add, Device_Atomic_Sub>;
template<class SystemTag>
using Atomic_Div = std::conditional_t<detail::is_host<SystemTag>,
		Host_Atomic_Add, Device_Atomic_Div>;
template<class SystemTag>
using Atomic_Mul = std::conditional_t<detail::is_host<SystemTag>,
		Host_Atomic_Add, Device_Atomic_Mul>;

#else //if __CUDACC__ is not defined

template<class SystemTag>
using Atomic_Add = std::enable_if_t<detail::is_host<SystemTag>,
		Host_Atomic_Add>;
template<class SystemTag>
using Atomic_Sub = std::enable_if_t<detail::is_host<SystemTag>,
		Host_Atomic_Add>;
template<class SystemTag>
using Atomic_Div = std::enable_if_t<detail::is_host<SystemTag>,
		Host_Atomic_Add>;
template<class SystemTag>
using Atomic_Mul = std::enable_if_t<detail::is_host<SystemTag>,
		Host_Atomic_Add>;


#endif

}
}

#undef BC_FORWARD_DEF
#undef BC_FORWARD_TO_APPLY

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

