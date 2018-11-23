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

namespace BC {
namespace et     {
namespace oper {

    struct negation {
        template<class lv> __BCinline__ lv operator ()(lv val) const {
            return -val;
        }
    };
    struct logical {
        template<class lv> __BCinline__ lv operator ()(lv val) const {
            return val == 0 ? 0 : 1;
        }
    };

}
}
}

#endif /* EXPRESSION_BINARY_FUNCTORS_H_ */

