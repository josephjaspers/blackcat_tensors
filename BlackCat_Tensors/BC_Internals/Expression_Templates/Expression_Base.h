/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_

#include "Internal_Type_Interface.h"
namespace BC {
namespace internal {

template<class derived>
struct Expression_Base
        : BC_internal_interface<derived>{
};

template<class derived>
struct Function_Interface
        : BC_internal_interface<derived> {
};

}
}

#endif /* EXPRESSION_BASE_H_ */
