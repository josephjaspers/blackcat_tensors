/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef STL_ITERATOR_COMMON_H_
#define STL_ITERATOR_COMMON_H_

namespace BC {
namespace module {
namespace stl {


enum direction {
    forward = 1,
    reverse = -1
};

struct scalar_access {

    template<class tensor_t>
    static auto& impl(tensor_t& tensor, int index) {
        return tensor.memptr()[index];
    }
};
}
}
}



#endif /* STL_ITERATOR_COMMON_H_ */
