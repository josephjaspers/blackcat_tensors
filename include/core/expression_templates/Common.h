/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_COMMON_H_
#define BC_EXPRESSION_TEMPLATES_COMMON_H_

#include <type_traits>

namespace BC {
namespace et {


template<class lv, class rv, class left = void>
struct dominant_type {
    BCINLINE static const auto& shape(const lv& l, const rv& r) {
        return l;
    }
};
template<class lv, class rv>
struct dominant_type<lv, rv, std::enable_if_t<(lv::DIMS < rv::DIMS)>> {

    BCINLINE static const auto& shape(const lv& l, const rv& r) {
        return r;
    }
};

//returns the class with the lower order rank
template<class lv, class rv, class left = void>
struct inferior_type {
    BCINLINE static const auto& shape(const lv& l, const rv& r) {
        return l;
    }
};
template<class lv, class rv>
struct inferior_type<lv, rv, std::enable_if_t<(lv::DIMS > rv::DIMS)>> {

    BCINLINE static const auto& shape(const lv& l, const rv& r) {
        return r;
    }
};


}
}


#endif /* BLACKCAT_COMPILERDEFINITIONS_H_ */
