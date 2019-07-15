/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef EXPRESSION_UNARY_FUNCTORS_H_
#define EXPRESSION_UNARY_FUNCTORS_H_

#include <cmath>
#include "Tags.h"

namespace BC {
namespace oper {

    struct negation {
        template<class lv> BCINLINE lv operator ()(lv val) const {
            return -val;
        }
        template<class lv> BCINLINE static lv apply(lv val) {
            return -val;
        }
    };

    template<class SystemTag, class ValueType>
    struct Sum;

    template<class ValueType>
    struct Sum<host_tag, ValueType> {
    	mutable ValueType total = 0;
    	mutable int index = 0;
    	template<class T>
    	auto operator ()(T&& value) const {
    		total += value;
    		std::cout << "current total" << total << " index " << index++ << std::endl;
    		return total;
    	}
    };

#ifdef __CUDACC__
    template<class ValueType> __device__
    struct Sum<device_tag, ValueType> {
    	mutable ValueType total = 0;

    	template<class T>
    	auto operator ()(T&& value) const {
			atomicAdd(&total, value);
			__syncthreads();
    		return total;
    	}
    };
#endif
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

