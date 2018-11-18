/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef TEMPORARY_ARRAY_H_
#define TEMPORARY_ARRAY_H_

namespace BC {
namespace et     {
namespace tree {

/*
 * used as a wrapper for BC_Arrays, enalbes the expression tree to traverse through and delete the correct
 */

template<class data_t>
struct temporary : data_t {
    using data_t::data_t;
};

}
}
}



#endif /* TEMPORARY_ARRAY_H_ */