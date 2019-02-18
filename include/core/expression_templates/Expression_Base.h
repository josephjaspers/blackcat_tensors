/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BC_EXPRESSION_TEMPLATES_EXPRESSION_BASE_H_
#define BC_EXPRESSION_TEMPLATES_EXPRESSION_BASE_H_

#include "Expression_Template_Base.h"


namespace BC {
namespace et {


template<class derived>
struct Expression_Base
        : Expression_Template_Base<derived>,
          BC_Expr {

	static constexpr bool copy_constructible = false;
	static constexpr bool move_constructible = false;
	static constexpr bool copy_assignable    = false;
	static constexpr bool move_assignable    = false;

};


}
}


#endif /* EXPRESSION_BASE_H_ */
