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
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

