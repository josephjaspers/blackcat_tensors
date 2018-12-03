/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTEE_TEMPORARY_H_
#define PTEE_TEMPORARY_H_

#include "Tree_Evaluator_Default.h"

namespace BC {
namespace et     {
namespace tree {

template<class T> struct evaluator<temporary<T>> :evaluator_default<temporary<T>> {

    static void deallocate_temporaries(temporary<T> tmp) {
        tmp.deallocate();
    }
};


}
}
}



#endif /* PTEE_TEMPORARY_H_ */
