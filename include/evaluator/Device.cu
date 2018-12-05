/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef BC_MATHEMATICS_DEVICE_H_
#define BC_MATHEMATICS_DEVICE_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Device_Impl.cu"

namespace BC {
namespace evaluator {

struct Device {

	 template<int d>
	    struct nd_evaluator_func {
	        struct n1 { template<class T> static void eval(T to) { gpu_impl::eval<<<blocks(to.size()),threads()>>>(to);   }};
	        struct n2 { template<class T> static void eval(T to) { gpu_impl::eval2d<<<blocks(to.size()),threads()>>>(to); }};
	        struct n3 { template<class T> static void eval(T to) { gpu_impl::eval3d<<<blocks(to.size()),threads()>>>(to); }};
	        struct n4 { template<class T> static void eval(T to) { gpu_impl::eval4d<<<blocks(to.size()),threads()>>>(to); }};
	        struct n5 { template<class T> static void eval(T to) { gpu_impl::eval5d<<<blocks(to.size()),threads()>>>(to); }};
	        using run = std::conditional_t<(d <= 1), n1,
	                        std::conditional_t< d == 2, n2,
	                            std::conditional_t< d == 3, n3,
	                                std::conditional_t< d == 4, n4, n5>>>>;

	        template<class T>
	        static void eval(T to) {
	            run::eval(to);
	            cudaDeviceSynchronize();
	        }
	    };

	    template<int d, class expr_t>
	    static void nd_evaluator(expr_t expr) {
	    	nd_evaluator_func<d>::eval(expr);
	    }

};

}
}

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
